using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;

[assembly: AssemblyTitle("MAA(AnchoredTorch)")]
[assembly: AssemblyDescription("Verify AnchoredTouch (1024) before launching MAA")]
[assembly: AssemblyProduct("MAA(AnchoredTorch)")]
[assembly: AssemblyCompany("Local")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class MaaAnchoredTorchLauncher
{
    private const string ExpectedProductVersion = "v6.16.8+5ee4315f7d9d79f28a3a76dd6a75eb452ac1ff66";
    private const string ExpectedMaaExeSha256 = "7377234bf379de7cf0d40612ce7eff0f88aa8a0dfb1913ac083e3b662e9bcc1e";
    private const string ExpectedMaaDllSha256 = "ec1b400b234dc03cd65cc14c3e881a4d7689ff968c36be421dcd9a312b4a6bcf";
    private const string ExpectedControlUnitSha256 = "6744c36a3e6e18630cc88224f4e7fc9d71a7eea482f8ec6875cb994cb81bf0e4";
    private const string ExpectedMouseMethod = "SendMessageWithWindowPos";

    [STAThread]
    private static int Main(string[] args)
    {
        bool checkOnly = Array.Exists(args, delegate(string value)
        {
            return String.Equals(value, "--check-only", StringComparison.OrdinalIgnoreCase);
        });
        string maaRoot = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string maaExe = Path.Combine(maaRoot, "MAA.exe");
        string maaDll = Path.Combine(maaRoot, "MAA.dll");
        string controlDll = Path.Combine(maaRoot, "MaaWin32ControlUnit.dll");
        string configPath = Path.Combine(maaRoot, "config", "gui.new.json");
        string backupManifest = Path.Combine(maaRoot, "codex-backups", "maa-pc-anchored-touch-v6.16.8", "manifest.json");
        string logPath = Path.Combine(maaRoot, "codex-backups", "anchored-touch-launcher.log");

        List<string> failures = new List<string>();
        List<string> warnings = new List<string>();

        RequireFile(maaExe, failures);
        RequireFile(maaDll, failures);
        RequireFile(controlDll, failures);
        RequireFile(configPath, failures);

        if (failures.Count == 0)
        {
            try
            {
                string productVersion = FileVersionInfo.GetVersionInfo(maaExe).ProductVersion ?? String.Empty;
                if (!String.Equals(productVersion, ExpectedProductVersion, StringComparison.Ordinal))
                {
                    failures.Add("检测到 MAA 版本变化：当前 " + productVersion + "，已验证版本 " + ExpectedProductVersion + "。");
                }
                if (!String.Equals(GetSha256(maaExe), ExpectedMaaExeSha256, StringComparison.Ordinal))
                {
                    failures.Add("MAA.exe 已被更新或修改。");
                }
                if (!String.Equals(GetSha256(maaDll), ExpectedMaaDllSha256, StringComparison.Ordinal))
                {
                    failures.Add("MAA.dll 校验不匹配，AnchoredTouch 映射可能已被更新覆盖。");
                }
                if (!String.Equals(GetSha256(controlDll), ExpectedControlUnitSha256, StringComparison.Ordinal))
                {
                    failures.Add("MaaWin32ControlUnit.dll 校验不匹配，AnchoredTouch 控制组件可能已被更新覆盖。");
                }

                string mouseMethod = GetConfiguredMouseMethod(configPath);
                if (!String.Equals(mouseMethod, ExpectedMouseMethod, StringComparison.Ordinal))
                {
                    failures.Add("当前鼠标方式为 '" + mouseMethod + "'，应为 '" + ExpectedMouseMethod + "'（运行时映射到 1024）。");
                }
            }
            catch (Exception ex)
            {
                failures.Add("读取 MAA 状态失败：" + ex.Message);
            }
        }

        if (!File.Exists(backupManifest))
        {
            warnings.Add("未找到补丁备份清单：" + backupManifest);
        }

        string status = failures.Count > 0 ? "BLOCKED" : (warnings.Count > 0 ? "PASS_WITH_WARNING" : "PASS");
        AppendLog(logPath, status, failures, warnings);

        if (failures.Count > 0)
        {
            if (checkOnly)
            {
                return 10;
            }
            string message = "AnchoredTouch 校验未通过，MAA 已被阻止启动。\r\n\r\n" +
                JoinItems(failures) +
                "\r\n\r\n这通常表示 MAA 已更新或补丁文件被替换。不要把旧版 DLL 强行覆盖到新版 MAA，请重新适配后再启动。" +
                "\r\n\r\n日志：" + logPath;
            MessageBox.Show(message, "MAA(AnchoredTorch)", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 10;
        }

        if (checkOnly)
        {
            return 0;
        }

        if (Process.GetProcessesByName("MAA").Length > 0)
        {
            MessageBox.Show("校验通过。MAA 已经在运行，因此没有重复启动。", "MAA(AnchoredTorch)", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 0;
        }

        try
        {
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = maaExe;
            startInfo.WorkingDirectory = maaRoot;
            startInfo.UseShellExecute = true;
            startInfo.Verb = "runas";
            Process.Start(startInfo);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show("校验通过，但启动 MAA 失败：" + ex.Message, "MAA(AnchoredTorch)", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 20;
        }
    }

    private static void RequireFile(string path, List<string> failures)
    {
        if (!File.Exists(path))
        {
            failures.Add("缺少必要文件：" + path);
        }
    }

    private static string GetSha256(string path)
    {
        using (SHA256 sha256 = SHA256.Create())
        using (FileStream stream = File.OpenRead(path))
        {
            byte[] hash = sha256.ComputeHash(stream);
            StringBuilder text = new StringBuilder(hash.Length * 2);
            foreach (byte value in hash)
            {
                text.Append(value.ToString("x2"));
            }
            return text.ToString();
        }
    }

    private static string GetConfiguredMouseMethod(string configPath)
    {
        JavaScriptSerializer serializer = new JavaScriptSerializer();
        Dictionary<string, object> root = serializer.Deserialize<Dictionary<string, object>>(File.ReadAllText(configPath, Encoding.UTF8));
        string profileName = root.ContainsKey("Current") ? Convert.ToString(root["Current"]) : "Default";
        if (String.IsNullOrEmpty(profileName))
        {
            profileName = "Default";
        }

        Dictionary<string, object> configurations = AsDictionary(root["Configurations"], "Configurations");
        if (!configurations.ContainsKey(profileName))
        {
            throw new InvalidDataException("Current MAA profile '" + profileName + "' was not found.");
        }
        Dictionary<string, object> profile = AsDictionary(configurations[profileName], profileName);
        Dictionary<string, object> gui = AsDictionary(profile["Gui"], "Gui");
        Dictionary<string, object> connectSettings = AsDictionary(gui["ConnectSettings"], "ConnectSettings");
        Dictionary<string, object> extras = AsDictionary(connectSettings["Extras"], "Extras");
        Dictionary<string, object> win32Extra = AsDictionary(extras["Win32Extra"], "Win32Extra");
        return Convert.ToString(win32Extra["MouseMethod"]);
    }

    private static Dictionary<string, object> AsDictionary(object value, string name)
    {
        Dictionary<string, object> dictionary = value as Dictionary<string, object>;
        if (dictionary == null)
        {
            throw new InvalidDataException("Invalid MAA configuration object: " + name);
        }
        return dictionary;
    }

    private static string JoinItems(List<string> items)
    {
        StringBuilder text = new StringBuilder();
        foreach (string item in items)
        {
            if (text.Length > 0)
            {
                text.Append("\r\n");
            }
            text.Append("• ");
            text.Append(item);
        }
        return text.ToString();
    }

    private static void AppendLog(string path, string status, List<string> failures, List<string> warnings)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            string line = DateTimeOffset.Now.ToString("o") + "\t" + status + "\t" +
                String.Join(" | ", failures.ToArray()) + "\t" + String.Join(" | ", warnings.ToArray()) + Environment.NewLine;
            File.AppendAllText(path, line, new UTF8Encoding(false));
        }
        catch
        {
            // Logging must not prevent a verified launch.
        }
    }
}
