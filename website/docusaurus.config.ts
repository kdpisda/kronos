import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// In production (GitHub Pages), the site lives at kdpisda.github.io/kronos/.
// In dev (`npm run start`), serve from the root so localhost:3030/ works.
const isProd = process.env.NODE_ENV === 'production';

const config: Config = {
  title: 'KRONOS',
  tagline: 'Plane-wave DFT, validated against Quantum ESPRESSO to meV precision.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: isProd ? 'https://kdpisda.github.io' : 'http://localhost:3030',
  baseUrl: isProd ? '/kronos/' : '/',

  organizationName: 'kdpisda',
  projectName: 'kronos',
  trailingSlash: false,

  // Existing markdown is plain text; H1-derived titles are fine for first run.
  // Switch to `warn` once we're certain all internal links resolve.
  onBrokenLinks: 'warn',

  // Existing docs/*.md contains C++ template syntax (`<vector>`, `<=`),
  // scientific notation (`|G|^2`), etc. that the MDX parser misreads as JSX.
  // Treat the existing files as plain CommonMark — no inline JSX.
  // Mermaid: ```mermaid fenced code blocks render as diagrams.
  markdown: {
    format: 'md',
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  // KaTeX CSS for math rendering. Loaded from CDN to avoid bundling ~200KB.
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPeEyROUyEMqkAB18oZW+zEdmsR7iYr1pgRRWVoEf3',
      crossorigin: 'anonymous',
    },
  ],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Docusaurus picks up partial files in `docs/` as pages; the existing
  // `docs/superpowers/` subtree (specs + plans) should NOT be served.
  // We also use the existing `docs/` at the repo root, NOT `website/docs/`.
  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          exclude: ['superpowers/**'],
          editUrl: 'https://github.com/kdpisda/kronos/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'KRONOS',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'mainSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          to: '/benchmarks',
          label: 'Benchmarks',
          position: 'left',
        },
        {
          href: 'https://github.com/kdpisda/kronos',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs/user_guide'},
            {label: 'Architecture', to: '/docs/architecture/overview'},
            {label: 'API Reference', to: '/docs/api_reference'},
            {label: 'Physics Notes', to: '/docs/physics_notes'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub Issues', href: 'https://github.com/kdpisda/kronos/issues'},
            {label: 'Discussions', href: 'https://github.com/kdpisda/kronos/discussions'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'Benchmarks', to: '/benchmarks'},
            {label: 'GitHub', href: 'https://github.com/kdpisda/kronos'},
            // rel="noopener" only (no noreferrer) so the author's own site
            // sees KRONOS docs as a traffic source in its analytics.
            {label: 'Author — kdpisda.in', href: 'https://kdpisda.in', rel: 'noopener'},
          ],
        },
      ],
      // The author's name in the copyright also links back, with the same
      // referrer-preserving rel. Docusaurus renders `copyright` as HTML.
      copyright: `Copyright © ${new Date().getFullYear()} <a href="https://kdpisda.in" target="_blank" rel="noopener">Kuldeep Pisda</a>. Licensed under GPL-3.0. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['cpp', 'cmake', 'yaml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
