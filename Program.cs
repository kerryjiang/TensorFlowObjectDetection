using System;
using System.Linq;
using System.IO;
using System.Net;
using System.Drawing;
using System.Threading.Tasks;
using SharpCompress.Common;
using SharpCompress.IO;
using SharpCompress.Readers;
using SkiaSharp;
using NumSharp;
using Tensorflow;

namespace TFPhoto
{
    class Program : Python
    {
        private static string _modelDir;
        private static string _imageDir;
        private static readonly string _pbFile = "frozen_inference_graph.pb";
        private static readonly string _labelFile = "mscoco_label_map.pbtxt";
        private const float MIN_SCORE = 0.5f;

        static async Task Main(string[] args)
        {
            await PrepareData();

            var p = new Program();
            p.Run();
        }

        private void Run()
        {
            
            // read in the input image
            var imgArr = ReadTensorFromImageFile(Path.Combine(_imageDir, "input.jpg"));

            var graph = new Graph().as_default();          
            graph.Import(Path.Join(_modelDir, _pbFile));

            Tensor tensorNum = graph.OperationByName("num_detections").output;
            Tensor tensorBoxes = graph.OperationByName("detection_boxes").output;
            Tensor tensorScores = graph.OperationByName("detection_scores").output;
            Tensor tensorClasses = graph.OperationByName("detection_classes").output;
            Tensor imgTensor = graph.OperationByName("image_tensor").output;

            Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

            with(tf.Session(graph), sess =>
            {
                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));
                
                NDArray[] resultArr = results.Data<NDArray>();
                
                BuildOutputImage(resultArr);
            });
        }

        private NDArray ReadTensorFromImageFile(string fileName)
        {
            return with(tf.Graph().as_default(), graph =>
            {
                var file_reader = tf.read_file(fileName, "file_reader");
                var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
                var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
                var dims_expander = tf.expand_dims(casted, 0);
                return with(tf.Session(graph), sess => sess.run(dims_expander));
            });
        }

        private void BuildOutputImage(NDArray[] resultArr)
        {            
            // get pbtxt items
            var pbTxtItems = PbtxtIteParser.ParsePbtxtFile(Path.Join(_modelDir, _labelFile));

            // get bitmap
            var bitmap = LoadBitmap(Path.Combine(_imageDir, "input.jpg"));
            
            float[] scores = resultArr[2].Data<float>();

            var canvas = new SKCanvas(bitmap);
            var paint = new SKPaint();
            paint.TextSize = 12.0f;
            paint.IsAntialias = true;
            paint.Color = new SKColor(0xFF, 0x00, 0x00);
            paint.IsStroke = true;
            paint.StrokeWidth = 1;
            paint.TextAlign = SKTextAlign.Center;

            for (int i = 0; i < scores.Length; i++)
            {
                float score = scores[i];

                if (score > MIN_SCORE)
                {
                    float[] boxes = resultArr[1].Data<float>();
                    float top = boxes[i * 4] * bitmap.Height;
                    float left = boxes[i * 4 + 1] * bitmap.Width;
                    float bottom = boxes[i * 4 + 2] * bitmap.Height;
                    float right = boxes[i * 4 + 3] * bitmap.Width;

                    Rectangle rect = new Rectangle()
                    {
                        X = (int)left,
                        Y = (int)top,
                        Width = (int)(right - left),
                        Height = (int)(bottom - top)
                    };

                    float[] ids = resultArr[3].Data<float>();

                    string name = pbTxtItems.Where(w => w.id == (int)ids[i]).Select(s => s.display_name).FirstOrDefault();

                    DrawObjectOnBitmap(canvas, rect, paint, score, name);
                }
            }

            canvas.Flush();

            var path = Path.Join(_imageDir, "output.jpg");

            using (var s = File.OpenWrite(path))
            {
                var d = SKImage.FromBitmap(bitmap).Encode(SKEncodedImageFormat.Jpeg, 100);
                d.SaveTo(s);
            }
            
            Console.WriteLine($"Processed image is saved as {path}");
        }

        private void DrawObjectOnBitmap(SKCanvas canvas, Rectangle rect, SKPaint paint, float score, string name)
        {
            canvas.DrawRect(rect.X, rect.Y, rect.Width, rect.Height, paint);

            string text = string.Format("{0}:{1}%", name, (int)(score * 100));
            canvas.DrawText(text, rect.Right + 5, rect.Top + 5, paint);
        }

        private static async Task PrepareData()
        {
            var modelDir = _modelDir = Path.Combine(AppContext.BaseDirectory, "data", "model");

            if (!Directory.Exists(modelDir))
                Directory.CreateDirectory(modelDir);

            var client = new WebClient();

            var localFilePath = Path.Combine(modelDir, "ssd_mobilenet_v1_coco.tar.gz");

            if (!File.Exists(localFilePath))
            {
                await client.DownloadFileTaskAsync("http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz", localFilePath);
                Unzip(localFilePath, modelDir);
            }                
            
            localFilePath = Path.Combine(modelDir, "mscoco_label_map.pbtxt");

            if (!File.Exists(localFilePath))
                await client.DownloadFileTaskAsync("https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt", localFilePath);

            _imageDir = Path.Combine(AppContext.BaseDirectory, "images");

            if (!Directory.Exists(_imageDir))
                Directory.CreateDirectory(_imageDir);

            localFilePath = Path.Combine(_imageDir, "input.jpg");

            if (!File.Exists(localFilePath))
                await client.DownloadFileTaskAsync("https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg", localFilePath);
        }

        private static void Unzip(string source, string saveTo)
        {
            using (var stream = File.OpenRead(source))
            using (var reader = ReaderFactory.Open(stream))
            {
                while (reader.MoveToNextEntry())
                {
                    if (!reader.Entry.IsDirectory)
                    {
                        reader.WriteEntryToDirectory(saveTo, new ExtractionOptions()
                        {
                            Overwrite = true
                        });
                    }
                }
            }
        }


        private SKBitmap LoadBitmap(string filePath)
        {
            return LoadBitmap(File.OpenRead(filePath), out SKEncodedOrigin orgin);
        }

        private SKBitmap LoadBitmap(Stream stream, out SKEncodedOrigin origin)
        {
            using (var s = new SKManagedStream(stream,  true))
            {
                using (var codec = SKCodec.Create(s))
                {
                    origin = codec.EncodedOrigin;
                    var info = codec.Info;
                    var bitmap = new SKBitmap(info.Width, info.Height, SKImageInfo.PlatformColorType, info.IsOpaque ? SKAlphaType.Opaque : SKAlphaType.Premul);

                    IntPtr length;
                    var result = codec.GetPixels(bitmap.Info, bitmap.GetPixels(out length));
                    if (result == SKCodecResult.Success || result == SKCodecResult.IncompleteInput)
                    {
                        return bitmap;
                    }
                    else
                    {
                        throw new ArgumentException("Unable to load bitmap from provided data");
                    }
                }
            }
        }
    }
}
