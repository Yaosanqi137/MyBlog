import { defineUserConfig } from "vuepress";
import { viteBundler } from '@vuepress/bundler-vite'

import {hopeTheme} from "vuepress-theme-hope";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "yao37的小博客",
  description: "yao37的小博客",

  bundler: viteBundler({
    viteOptions: {
        server: {
            allowedHosts: ['gstech.fun']
        }
    }
  }),
  // 和 PWA 一起启用
  // shouldPrefetch: false,

    theme: hopeTheme({
        markdown: {
            math: {
                type: "katex",
            },
        },
    }),
});
