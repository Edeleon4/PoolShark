using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Drawing;
using Mono.Options;


namespace apiCall_wildNature
{
    class Program
    {
        //----------------------
        //parameters
        //----------------------
        static int imgMaxWidth = 1500;
        static string outSubdir = "outputs/";
        static bool drawRectangle = true;
        static bool skipIfAlreadyProcessed = true;
        static string apiUrl = "http://giraffeDetection.azurewebsites.net/api";
        static string imgDir = null;
        //static string apiUrl = "http://localhost:58018/api";


        //----------------------
        //helper functions
        //----------------------
        private static Image cropImage(Image img, Rectangle cropArea)
        {
            Bitmap bmpImage = new Bitmap(img);
            return bmpImage.Clone(cropArea, bmpImage.PixelFormat);
        }
        
        static void showUsage(OptionSet p)
        {
            Console.WriteLine("Command line usage:");
            Console.WriteLine(" Example: apiCall_wildNature --imgDir=./images/ [--imgMaxWidth=1500]");
            Console.WriteLine(" Options:");
            p.WriteOptionDescriptions(Console.Out);
            Environment.Exit(-1);
        }

        static Image applyExifRotation(Image img)
        {
            if (Array.IndexOf(img.PropertyIdList, 274) > -1)
            {
                var orientation = (int)img.GetPropertyItem(274).Value[0];
                switch (orientation)
                {
                    case 1:
                        break;
                    case 2:
                        img.RotateFlip(RotateFlipType.RotateNoneFlipX);
                        break;
                    case 3:
                        img.RotateFlip(RotateFlipType.Rotate180FlipNone);
                        break;
                    case 4:
                        img.RotateFlip(RotateFlipType.Rotate180FlipX);
                        break;
                    case 5:
                        img.RotateFlip(RotateFlipType.Rotate90FlipX);
                        break;
                    case 6:
                        img.RotateFlip(RotateFlipType.Rotate90FlipNone);
                        break;
                    case 7:
                        img.RotateFlip(RotateFlipType.Rotate270FlipX);
                        break;
                    case 8:
                        img.RotateFlip(RotateFlipType.Rotate270FlipNone);
                        break;
                }
                img.RemovePropertyItem(274); //exif data is invalid now, hence remove
            }
            return img;
        }


//----------------------
//main
//----------------------
static void Main(string[] args)
        {
            //parse command line arguments
            //List<string> extra;
            var argsOptionSet = new OptionSet()
            {
                { "i|imgDir=", "Image {DIRECTORY}", s => imgDir=s },
                { "m|imgMaxWidth=", "Maximum image {WIDTH} in pixels (integer) for API call (default: " + imgMaxWidth + ").", s => imgMaxWidth = Convert.ToInt32(s)},
                { "r|drawRectangle=", "Draw image with detection rectangle (default: " + drawRectangle + ").", s => drawRectangle = Convert.ToBoolean(s)},
                { "s|skipIfAlreadyProcessed=", " (default: " + skipIfAlreadyProcessed + ").", s => skipIfAlreadyProcessed = Convert.ToBoolean(s)},
                { "u|apiUrl=", "Rest Api {URL} (default: " + apiUrl + ").", s => apiUrl=s }
            };
            try
            {
                argsOptionSet.Parse(args);
            }
            catch (OptionException e)
            {
                showUsage(argsOptionSet);
            }
            if (imgDir == null)
            {
                showUsage(argsOptionSet);
            }

            //display parameters
            Console.WriteLine("Parameters:");
            Console.WriteLine("  Image directory = " + imgDir);
            Console.WriteLine("  Maximum image width [pixels] = " + imgMaxWidth);
            Console.WriteLine("  Api Url = " + apiUrl);

            //init
            imgDir = imgDir.Replace('\\', '/') + '/';
            Console.WriteLine("Start time " + DateTime.Now.ToString("HH:mm:ss"));
            var outDir = imgDir + outSubdir;
            Directory.CreateDirectory(outDir);
            var httpClient = new HttpClient();

            //loop over all images in the directory
            var imgFilenames = Directory.EnumerateFiles(imgDir) .Select(p => Path.GetFileName(p).ToLower())
                .Where(s => s.EndsWith(".jpg") || s.EndsWith(".jpeg") || s.EndsWith(".gif") || s.EndsWith(".tif")
                            || s.EndsWith(".tiff") || s.EndsWith(".bmp") || s.EndsWith(".png"));
            foreach (var imgFilename in imgFilenames)
            {
                Console.WriteLine("\nProcessing image " + imgFilename + "(" + DateTime.Now.ToString("HH:mm:ss") + ")");
                String responseString = "";
                try
                {
                    string cropPath = outDir + imgFilename + ".crop.jpg";
                    if (skipIfAlreadyProcessed && File.Exists(cropPath))
                    {
                        Console.WriteLine("Skipping image " + imgFilename + " since crop exists already: " + cropPath);
                        continue;
                    }


                    //load image and process exif flag
                    var imgPath = imgDir + imgFilename;
                    var origImg = Bitmap.FromFile(imgPath);
                    origImg = applyExifRotation(origImg);

                    //get byte stream from (potentially rotated) image
                    var tmpFilename = "tmp_" + Path.GetRandomFileName();
                    origImg.Save(tmpFilename);
                    var imgBytes = File.ReadAllBytes(tmpFilename);
                    File.Delete(tmpFilename);

                    //downscale
                    var scaleFactor = Math.Min(1.0, 1.0 * imgMaxWidth / origImg.Width);
                    if (scaleFactor < 1.0)
                    {
                        var resizedImg = new Bitmap(origImg,
                            new Size(Convert.ToInt32(origImg.Width * scaleFactor), Convert.ToInt32(origImg.Height * scaleFactor)));
                        Console.WriteLine("Downsizing image for API call to: " + resizedImg.Size);
                        //Console.WriteLine("Calling API with downscaled image by factor=" + scaleFactor);
                        resizedImg.Save(tmpFilename);
                        imgBytes = File.ReadAllBytes(tmpFilename);
                        File.Delete(tmpFilename);
                    }

                    //call API 
                    Console.WriteLine("Calling REST api...");
                    var requestContent = new MultipartFormDataContent();
                    var fileContent = new ByteArrayContent(imgBytes);
                    fileContent.Headers.ContentDisposition = new ContentDispositionHeaderValue("attachment")
                    {
                        Name = "\"file\"",
                        FileName = "\"" + "AAA" + imgFilename + "\""
                    };
                    fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse("multipart/form-data");
                    requestContent.Add(fileContent);
                    responseString = httpClient.PostAsync(apiUrl, requestContent).Result.Content.ReadAsStringAsync().Result;

                    //parse result
                    Console.WriteLine("Parsing REST api output...");
                    //Console.WriteLine("responseString = " + responseString);
                    var apiOutput = JsonConvert.DeserializeObject<dynamic>(responseString);
                    var errorMsg = apiOutput["error"].Value;
                    if (errorMsg != "")
                        throw new Exception(errorMsg);
                    var boGiraffeFound = apiOutput["boGiraffeFound"].Value == "True";
                    var confidence = Convert.ToSingle(apiOutput["confidence"].Value);
                    //string debugLog = apiOutput["debugLog"].Value;
                    var processingTimeMs = Convert.ToDouble(apiOutput["processingTimeMs"].Value);
                    int left = 0, top = 0, right = 0, bottom = 0;
                    if (boGiraffeFound)
                    {
                        left = Convert.ToInt32(Convert.ToDouble(apiOutput["left"].Value) / scaleFactor);
                        top = Convert.ToInt32(Convert.ToDouble(apiOutput["top"].Value) / scaleFactor);
                        right = Convert.ToInt32(Convert.ToDouble(apiOutput["right"].Value) / scaleFactor);
                        bottom = Convert.ToInt32(Convert.ToDouble(apiOutput["bottom"].Value) / scaleFactor);
                    }

                    //print result
                    //Console.WriteLine("boGiraffeFound = " + boGiraffeFound);
                    //Console.WriteLine("debugLog = " + debugLog.Replace("<br>","\n"));
                    Console.WriteLine("Giraffe body found with confidence " + Convert.ToSingle(Math.Round(confidence, 3)) + " at co-ordinates:");
                    Console.WriteLine("  Left = " + left);
                    Console.WriteLine("  Top = " + top);
                    Console.WriteLine("  Right = " + right);
                    Console.WriteLine("  Bottom = " + bottom);
                    Console.WriteLine("Processing time on server [ms] = " + processingTimeMs);

                    //save result
                    var coords = left + "," + top + "," + right + "," + bottom + "," + confidence;
                    File.WriteAllText(outDir + imgFilename + ".detection.csv", coords);

                    //crop image
                    var rect = new Rectangle(left, top, right - left, bottom - top);
                    var imgCropped = cropImage(origImg, rect);
                    imgCropped.Save(outDir + imgFilename + ".crop.jpg");

                    //draw rectangle and confidence 
                    if (drawRectangle)
                    {
                        string confidenceString = confidence.ToString("00.0");
                        using (Graphics g = Graphics.FromImage(origImg))
                        {
                            int thickness = (int)Math.Round(0.03 * 0.5 * (origImg.Width + origImg.Height));
                            g.DrawRectangle(new Pen(Color.Blue, thickness), rect);
                            Font drawFont = new System.Drawing.Font("Arial", thickness * 5);
                            SolidBrush drawBrush = new System.Drawing.SolidBrush(System.Drawing.Color.Blue);
                            g.DrawString(confidenceString, drawFont, drawBrush, 10, 10);
                        }

                        var scaleFactorRectImg = 800.0 / Math.Max(origImg.Width, origImg.Height);
                        origImg = new Bitmap(origImg,
                           new Size(Convert.ToInt32(origImg.Width * scaleFactorRectImg), Convert.ToInt32(origImg.Height * scaleFactorRectImg)));
                        origImg.Save(outDir + imgFilename + ".rect.jpg");
                        origImg.Save(outDir + confidenceString + "_" + imgFilename + ".rect.jpg");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Program exiting with error: " + ex.Message);
                    Console.WriteLine("Inner exception = " + ex.InnerException);
                    Console.WriteLine("responseString = " + responseString);
                }
            }

            Console.WriteLine("Press return to exit ...");
            Console.ReadLine();
        }
    }
}
