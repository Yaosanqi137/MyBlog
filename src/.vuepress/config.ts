import { defineUserConfig } from "vuepress";
import { viteBundler } from '@vuepress/bundler-vite'
import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "博客演示",
  description: "vuepress-theme-hope 的博客演示",

  theme,
  bundler: viteBundler({
    viteOptions: {
        server: {
            allowedHosts: ['gstech.fun']
        }
    }
  }),
  // 和 PWA 一起启用
  // shouldPrefetch: false,
});
