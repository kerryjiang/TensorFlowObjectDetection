using System.Collections.Generic;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace TFPhoto
{
    public static class PbtxtIteParser
    {
        public static List<PbtxtItem> ParsePbtxtFile(string filePath)
        {
            var line = string.Empty;

            var list = new List<PbtxtItem>();

            using (var reader = new StreamReader(filePath))
            {
                var sb = new StringBuilder();
                
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();

                    if (sb.Length == 0) // first line
                    {
                        var pos = line.IndexOf('{');

                        if (pos >= 0)
                            sb.AppendLine(line.Substring(pos));
                        else
                            sb.AppendLine();
                        
                        continue;
                    }

                    if (line.Length == 1 && line[0] == '}')
                    {
                        sb.AppendLine(line);
                        list.Add(JsonConvert.DeserializeObject<PbtxtItem>(sb.ToString()));
                        sb.Clear();
                        continue;
                    }

                    sb.AppendLine(line + ",");
                }
            }

            return list;
        }
    }
}