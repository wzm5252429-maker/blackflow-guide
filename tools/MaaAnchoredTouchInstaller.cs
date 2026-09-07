using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;

[assembly: AssemblyTitle("MAA AnchoredTouch 安装程序")]
[assembly: AssemblyDescription("MAA PC v6.16.8 AnchoredTouch compatibility installer")]
[assembly: AssemblyProduct("MAA AnchoredTouch")]
[assembly: AssemblyCompany("Community compatibility package")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class MaaAnchoredTouchInstaller
{
    private const string ExpectedVersion = "v6.16.8+5ee4315f7d9d79f28a3a76dd6a75eb452ac1ff66";
    private const string ExpectedExeHash = "7377234bf379de7cf0d40612ce7eff0f88aa8a0dfb1913ac083e3b662e9bcc1e";
    private const string OriginalMaaDllHash = "94050652d294ff36a756ad8389e8e20992d3c29403c952cb1e1ccaf105ef3f5a";
    private const string PatchedMaaDllHash = "ec1b400b234dc03cd65cc14c3e881a4d7689ff968c36be421dcd9a312b4a6bcf";
    private const string OriginalControlHash = "a47e5364305aa0d40c3720d6486b8ddd6215b24dce99717de53344a47c9b4805";
    private const string PatchedControlHash = "6744c36a3e6e18630cc88224f4e7fc9d71a7eea482f8ec6875cb994cb81bf0e4";
    private const string LauncherHash = "54d426380300cb10e85abf6c628386e9172fba3961ce85f64664edf55a3f8e92";
    private const string MouseMethod = "SendMessageWithWindowPos";

    [STAThread]
    private static int Main(string[] args)
    {
        bool silent = HasArg(args, "--silent");
        bool noShortcuts = HasArg(args, "--no-shortcuts");
        bool allowRunningForTest = HasArg(args, "--allow-running-for-test");
        string target = GetArg(args, "--target");
        try
        {
            if (String.IsNullOrWhiteSpace(target)) target = DetectMaaDirectory();
            if (String.IsNullOrWhiteSpace(target) && !silent) target = ChooseMaaDirectory();
            if (String.IsNullOrWhiteSpace(target)) return Fail("没有找到兼容的 MAA。请选择新版 MAA.exe 所在目录。", silent, 2);

            target = Path.GetFullPath(target.Trim().Trim('"')).TrimEnd(Path.DirectorySeparatorChar);
            string reason;
            if (!ValidateTarget(target, out reason)) return Fail(reason, silent, 3);

            if (!allowRunningForTest && Process.GetProcessesByName("MAA").Length != 0)
                return Fail("MAA 正在运行。请退出 MAA 后重新运行安装包。", silent, 4);

            if (!silent)
            {
                DialogResult answer = MessageBox.Show(
                    "已找到兼容的 MAA：\r\n" + target +
                    "\r\n\r\n将安装 AnchoredTouch 补丁、校验启动器以及桌面/开始菜单入口。原文件会自动备份。",
                    "MAA AnchoredTouch 安装程序", MessageBoxButtons.OKCancel, MessageBoxIcon.Information);
                if (answer != DialogResult.OK) return 1;
            }

            Install(target, noShortcuts);
            if (!silent)
            {
                MessageBox.Show(
                    "安装完成。\r\n\r\n以后请使用“MAA(AnchoredTorch)”启动。\r\n" +
                    "游戏窗口保持 16:9 且不要最小化；MAA 界面可以最小化。\r\n\r\n安装目录：" + target,
                    "MAA AnchoredTouch", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            return 0;
        }
        catch (UnauthorizedAccessException)
        {
            return Fail("没有写入 MAA 目录的权限。请右键安装包，选择“以管理员身份运行”。", silent, 5);
        }
        catch (Exception ex)
        {
            return Fail("安装失败，未完成的文件不会被当作有效安装：\r\n" + ex.Message, silent, 10);
        }
    }

    private static void Install(string root, bool noShortcuts)
    {
        string maaDll = Path.Combine(root, "MAA.dll");
        string controlDll = Path.Combine(root, "MaaWin32ControlUnit.dll");
        string config = Path.Combine(root, "config", "gui.new.json");
        string launcher = Path.Combine(root, "MAA(AnchoredTorch).exe");
        string backup = Path.Combine(root, "codex-backups", "maa-pc-anchored-touch-v6.16.8");
        Directory.CreateDirectory(backup);

        string maaHash = Sha256(maaDll);
        string controlHash = Sha256(controlDll);
        BackupOnce(maaDll, Path.Combine(backup, "MAA.dll.original"), OriginalMaaDllHash, PatchedMaaDllHash);
        BackupOnce(controlDll, Path.Combine(backup, "MaaWin32ControlUnit.dll.original"), OriginalControlHash, PatchedControlHash);
        string backupConfig = Path.Combine(backup, "gui.new.json.original");
        BackupConfigOnce(config, backupConfig);
        string originalMethod = GetMouseMethod(backupConfig);

        WriteResourceAtomically("Payload.MAA.dll", maaDll, PatchedMaaDllHash);
        WriteResourceAtomically("Payload.MaaWin32ControlUnit.dll", controlDll, PatchedControlHash);
        WriteResourceAtomically("Payload.Launcher.exe", launcher, LauncherHash);
        SetMouseMethod(config, MouseMethod);

        string manifest = "{\r\n" +
            "  \"created_at\": \"" + DateTimeOffset.Now.ToString("o") + "\",\r\n" +
            "  \"maa_version\": \"" + ExpectedVersion + "\",\r\n" +
            "  \"maa_dll_sha256\": \"" + (maaHash == PatchedMaaDllHash ? OriginalMaaDllHash : maaHash) + "\",\r\n" +
            "  \"control_unit_sha256\": \"" + (controlHash == PatchedControlHash ? OriginalControlHash : controlHash) + "\",\r\n" +
            "  \"original_mouse_method\": \"" + JsonEscape(originalMethod) + "\",\r\n" +
            "  \"framework_payload\": \"v5.13.0-beta.5\",\r\n" +
            "  \"installer\": \"portable-v1\"\r\n" +
            "}\r\n";
        File.WriteAllText(Path.Combine(backup, "manifest.json"), manifest, new UTF8Encoding(false));

        if (!noShortcuts)
        {
            CreateShortcut(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "MAA(AnchoredTorch).lnk"), launcher, root);
            string programs = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
            CreateShortcut(Path.Combine(programs, "MAA(AnchoredTorch).lnk"), launcher, root);
        }

        if (Sha256(maaDll) != PatchedMaaDllHash || Sha256(controlDll) != PatchedControlHash ||
            Sha256(launcher) != LauncherHash || GetMouseMethod(config) != MouseMethod)
            throw new InvalidDataException("安装后的完整性校验未通过。");
    }

    private static void BackupOnce(string source, string destination, string originalHash, string patchedHash)
    {
        if (File.Exists(destination))
        {
            if (Sha256(destination) != originalHash) throw new InvalidDataException("已有备份校验失败：" + destination);
            return;
        }
        string current = Sha256(source);
        if (current == patchedHash) throw new InvalidDataException("目标已打补丁但缺少原始备份，已拒绝覆盖：" + source);
        if (current != originalHash) throw new InvalidDataException("原文件版本不匹配，已拒绝覆盖：" + source);
        File.Copy(source, destination, false);
    }

    private static void BackupConfigOnce(string source, string destination)
    {
        if (!File.Exists(destination)) File.Copy(source, destination, false);
    }

    private static void WriteResourceAtomically(string resourceName, string destination, string expectedHash)
    {
        string temp = destination + ".anchoredtouch-new";
        using (Stream input = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
        {
            if (input == null) throw new InvalidDataException("安装包缺少资源：" + resourceName);
            using (FileStream output = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None)) input.CopyTo(output);
        }
        if (Sha256(temp) != expectedHash) { File.Delete(temp); throw new InvalidDataException("安装包资源校验失败：" + resourceName); }
        File.Copy(temp, destination, true);
        File.Delete(temp);
    }

    private static bool ValidateTarget(string root, out string reason)
    {
        string exe = Path.Combine(root, "MAA.exe");
        string maaDll = Path.Combine(root, "MAA.dll");
        string control = Path.Combine(root, "MaaWin32ControlUnit.dll");
        string config = Path.Combine(root, "config", "gui.new.json");
        if (!File.Exists(exe) || !File.Exists(maaDll) || !File.Exists(control))
        { reason = "所选目录不是受支持的新版 MAA PC 目录（缺少 MAA.exe、MAA.dll 或 MaaWin32ControlUnit.dll）。"; return false; }
        if (!File.Exists(config))
        { reason = "尚未找到 config\\gui.new.json。请先正常启动并退出一次 MAA，再运行安装包。"; return false; }
        string version = FileVersionInfo.GetVersionInfo(exe).ProductVersion ?? "";
        if (version != ExpectedVersion || Sha256(exe) != ExpectedExeHash)
        { reason = "MAA 版本不兼容。此安装包仅支持 " + ExpectedVersion + "，检测到：" + version + "。为避免损坏，未修改任何文件。"; return false; }
        string mh = Sha256(maaDll), ch = Sha256(control);
        if (mh != OriginalMaaDllHash && mh != PatchedMaaDllHash)
        { reason = "MAA.dll 与已验证版本不匹配，未修改任何文件。"; return false; }
        if (ch != OriginalControlHash && ch != PatchedControlHash)
        { reason = "MaaWin32ControlUnit.dll 与已验证版本不匹配，未修改任何文件。"; return false; }
        reason = ""; return true;
    }

    private static string DetectMaaDirectory()
    {
        List<string> candidates = new List<string>();
        AddCandidate(candidates, AppDomain.CurrentDomain.BaseDirectory);
        AddCandidate(candidates, Environment.CurrentDirectory);
        foreach (Process p in Process.GetProcessesByName("MAA"))
        {
            try { AddCandidate(candidates, Path.GetDirectoryName(p.MainModule.FileName)); } catch { }
        }
        AddShortcutCandidates(candidates, Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory));
        AddShortcutCandidates(candidates, Environment.GetFolderPath(Environment.SpecialFolder.Programs));
        AddCandidate(candidates, @"D:\明日方舟");
        foreach (string candidate in candidates)
        {
            string ignored;
            try { if (ValidateTarget(candidate, out ignored)) return candidate; } catch { }
        }
        return null;
    }

    private static void AddCandidate(List<string> list, string path)
    {
        if (String.IsNullOrWhiteSpace(path)) return;
        path = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar);
        if (!list.Exists(delegate(string x) { return String.Equals(x, path, StringComparison.OrdinalIgnoreCase); })) list.Add(path);
    }

    private static void AddShortcutCandidates(List<string> list, string directory)
    {
        try
        {
            if (!Directory.Exists(directory)) return;
            Type type = Type.GetTypeFromProgID("WScript.Shell");
            object shell = Activator.CreateInstance(type);
            foreach (string link in Directory.GetFiles(directory, "*.lnk", SearchOption.AllDirectories))
            {
                try
                {
                    object shortcut = type.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { link });
                    string target = Convert.ToString(shortcut.GetType().InvokeMember("TargetPath", BindingFlags.GetProperty, null, shortcut, null));
                    if (String.Equals(Path.GetFileName(target), "MAA.exe", StringComparison.OrdinalIgnoreCase) ||
                        String.Equals(Path.GetFileName(target), "MAA(AnchoredTorch).exe", StringComparison.OrdinalIgnoreCase))
                        AddCandidate(list, Path.GetDirectoryName(target));
                }
                catch { }
            }
        }
        catch { }
    }

    private static string ChooseMaaDirectory()
    {
        using (OpenFileDialog dialog = new OpenFileDialog())
        {
            dialog.Title = "请选择新版 MAA.exe";
            dialog.Filter = "MAA 主程序 (MAA.exe)|MAA.exe";
            dialog.CheckFileExists = true;
            return dialog.ShowDialog() == DialogResult.OK ? Path.GetDirectoryName(dialog.FileName) : null;
        }
    }

    private static string GetMouseMethod(string configPath)
    {
        Dictionary<string, object> root = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(File.ReadAllText(configPath, Encoding.UTF8));
        string current = root.ContainsKey("Current") ? Convert.ToString(root["Current"]) : "Default";
        if (String.IsNullOrEmpty(current)) current = "Default";
        Dictionary<string, object> configurations = Dict(root["Configurations"]);
        Dictionary<string, object> profile = Dict(configurations[current]);
        Dictionary<string, object> gui = Dict(profile["Gui"]);
        Dictionary<string, object> connect = Dict(gui["ConnectSettings"]);
        Dictionary<string, object> extras = Dict(connect["Extras"]);
        Dictionary<string, object> win32 = Dict(extras["Win32Extra"]);
        return Convert.ToString(win32["MouseMethod"]);
    }

    private static void SetMouseMethod(string configPath, string value)
    {
        JavaScriptSerializer serializer = new JavaScriptSerializer();
        serializer.MaxJsonLength = Int32.MaxValue;
        Dictionary<string, object> root = serializer.Deserialize<Dictionary<string, object>>(File.ReadAllText(configPath, Encoding.UTF8));
        string current = root.ContainsKey("Current") ? Convert.ToString(root["Current"]) : "Default";
        if (String.IsNullOrEmpty(current)) current = "Default";
        Dictionary<string, object> configurations = Dict(root["Configurations"]);
        Dictionary<string, object> profile = Dict(configurations[current]);
        Dictionary<string, object> gui = Dict(profile["Gui"]);
        Dictionary<string, object> connect = Dict(gui["ConnectSettings"]);
        Dictionary<string, object> extras = Dict(connect["Extras"]);
        Dictionary<string, object> win32 = Dict(extras["Win32Extra"]);
        win32["MouseMethod"] = value;
        File.WriteAllText(configPath, serializer.Serialize(root), new UTF8Encoding(false));
    }

    private static Dictionary<string, object> Dict(object value)
    {
        Dictionary<string, object> result = value as Dictionary<string, object>;
        if (result == null) throw new InvalidDataException("MAA 配置结构不兼容。");
        return result;
    }

    private static void CreateShortcut(string linkPath, string target, string workingDirectory)
    {
        Type type = Type.GetTypeFromProgID("WScript.Shell");
        object shell = Activator.CreateInstance(type);
        object shortcut = type.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { linkPath });
        Type shortcutType = shortcut.GetType();
        shortcutType.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { target });
        shortcutType.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { workingDirectory });
        shortcutType.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { Path.Combine(workingDirectory, "MAA.exe") + ",0" });
        shortcutType.InvokeMember("Description", BindingFlags.SetProperty, null, shortcut, new object[] { "MAA AnchoredTouch 校验启动器" });
        shortcutType.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
    }

    private static string Sha256(string path)
    {
        using (SHA256 sha = SHA256.Create()) using (FileStream input = File.OpenRead(path))
        {
            byte[] hash = sha.ComputeHash(input); StringBuilder result = new StringBuilder(64);
            foreach (byte b in hash) result.Append(b.ToString("x2")); return result.ToString();
        }
    }

    private static bool HasArg(string[] args, string name)
    { return Array.Exists(args, delegate(string x) { return String.Equals(x, name, StringComparison.OrdinalIgnoreCase); }); }

    private static string GetArg(string[] args, string name)
    {
        for (int i = 0; i + 1 < args.Length; i++) if (String.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }

    private static int Fail(string message, bool silent, int code)
    {
        if (!silent) MessageBox.Show(message, "MAA AnchoredTouch 安装程序", MessageBoxButtons.OK, MessageBoxIcon.Error);
        else Console.Error.WriteLine(message);
        return code;
    }

    private static string JsonEscape(string value)
    { return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\""); }
}
