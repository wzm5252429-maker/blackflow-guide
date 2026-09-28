import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "黑流树海路线参谋",
  description:
    "黑流树海神经网络自动执行：连接本机后，一键启动一结局非战斗路线辅助。含能力说明、节点图鉴与作战资料。",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">{children}</body>
    </html>
  );
}
