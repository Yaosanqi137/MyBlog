import { defineUserConfig } from "vuepress";
import { viteBundler } from '@vuepress/bundler-vite'
import theme from "./theme.js";

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

    theme,
    head: [
        [
            "script",
            {},
            `
            <!-- Cloudflare Web Analytics -->
            <script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "2b0f222eb4b7437f9447c9b911c6d590"}'></script>
            <!-- End Cloudflare Web Analytics -->
            `,
        ]
    ]
});
