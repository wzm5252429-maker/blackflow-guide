using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

internal static class RegisterMaaAnchoredTorchShortcut
{
    private static readonly Guid AppUserModelFormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length != 4)
        {
            Console.Error.WriteLine("Usage: register-shortcut <link> <target> <working-dir> <icon>");
            return 2;
        }

        string linkPath = args[0];
        string targetPath = args[1];
        string workingDirectory = args[2];
        string iconPath = args[3];
        string appId = targetPath;

        IShellLinkW shellLink = (IShellLinkW)new ShellLink();
        ThrowIfFailed(shellLink.SetPath(targetPath));
        ThrowIfFailed(shellLink.SetWorkingDirectory(workingDirectory));
        ThrowIfFailed(shellLink.SetDescription("校验 AnchoredTouch (1024) 后启动 MAA"));
        ThrowIfFailed(shellLink.SetIconLocation(iconPath, 0));
        ThrowIfFailed(shellLink.SetShowCmd(1));

        IPropertyStore propertyStore = (IPropertyStore)shellLink;
        SetString(propertyStore, new PropertyKey(AppUserModelFormatId, 2), Quote(targetPath));
        SetString(propertyStore, new PropertyKey(AppUserModelFormatId, 3), iconPath + ",0");
        SetString(propertyStore, new PropertyKey(AppUserModelFormatId, 5), appId);
        ThrowIfFailed(propertyStore.Commit());

        IPersistFile persistFile = (IPersistFile)shellLink;
        persistFile.Save(linkPath, true);

        SHChangeNotify(0x00000002, 0x0005, linkPath, null);
        SHChangeNotify(0x08000000, 0, null, null);
        return 0;
    }

    private static string Quote(string value)
    {
        return "\"" + value + "\"";
    }

    private static void SetString(IPropertyStore store, PropertyKey key, string value)
    {
        PropVariant variant = PropVariant.FromString(value);
        try
        {
            ThrowIfFailed(store.SetValue(ref key, ref variant));
        }
        finally
        {
            PropVariantClear(ref variant);
        }
    }

    private static void ThrowIfFailed(int result)
    {
        if (result < 0)
        {
            Marshal.ThrowExceptionForHR(result);
        }
    }

    [DllImport("ole32.dll")]
    private static extern int PropVariantClear(ref PropVariant variant);

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern void SHChangeNotify(uint eventId, uint flags, string item1, string item2);

    [ComImport]
    [Guid("00021401-0000-0000-C000-000000000046")]
    private class ShellLink
    {
    }

    [ComImport]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [Guid("000214F9-0000-0000-C000-000000000046")]
    private interface IShellLinkW
    {
        [PreserveSig] int GetPath(IntPtr file, int maxPath, IntPtr findData, uint flags);
        [PreserveSig] int GetIDList(out IntPtr idList);
        [PreserveSig] int SetIDList(IntPtr idList);
        [PreserveSig] int GetDescription(IntPtr name, int maxName);
        [PreserveSig] int SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        [PreserveSig] int GetWorkingDirectory(IntPtr directory, int maxPath);
        [PreserveSig] int SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string directory);
        [PreserveSig] int GetArguments(IntPtr arguments, int maxPath);
        [PreserveSig] int SetArguments([MarshalAs(UnmanagedType.LPWStr)] string arguments);
        [PreserveSig] int GetHotkey(out short hotkey);
        [PreserveSig] int SetHotkey(short hotkey);
        [PreserveSig] int GetShowCmd(out int showCommand);
        [PreserveSig] int SetShowCmd(int showCommand);
        [PreserveSig] int GetIconLocation(IntPtr iconPath, int iconPathLength, out int iconIndex);
        [PreserveSig] int SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string iconPath, int iconIndex);
        [PreserveSig] int SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        [PreserveSig] int Resolve(IntPtr window, uint flags);
        [PreserveSig] int SetPath([MarshalAs(UnmanagedType.LPWStr)] string path);
    }

    [ComImport]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    private interface IPropertyStore
    {
        [PreserveSig] int GetCount(out uint propertyCount);
        [PreserveSig] int GetAt(uint propertyIndex, out PropertyKey key);
        [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
        [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
        [PreserveSig] int Commit();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct PropertyKey
    {
        public Guid FormatId;
        public uint PropertyId;

        public PropertyKey(Guid formatId, uint propertyId)
        {
            FormatId = formatId;
            PropertyId = propertyId;
        }
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct PropVariant
    {
        [FieldOffset(0)] public ushort VariantType;
        [FieldOffset(8)] public IntPtr PointerValue;

        public static PropVariant FromString(string value)
        {
            PropVariant variant = new PropVariant();
            variant.VariantType = 31;
            variant.PointerValue = Marshal.StringToCoTaskMemUni(value);
            return variant;
        }
    }
}
