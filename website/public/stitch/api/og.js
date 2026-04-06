import { ImageResponse } from '@vercel/og';
import React from 'react';

export const config = { runtime: 'edge' };

const h = (type, props, ...children) =>
  React.createElement(type, props, ...children);

export default function handler(req) {
  const url = new URL(req.url);
  const title = url.searchParams.get('title') || 'Boundary Phase Resonance';
  const sub   = url.searchParams.get('sub')   || 'Open Research Framework';

  return new ImageResponse(
    h('div', {
      style: {
        height: '100%', width: '100%',
        display: 'flex', flexDirection: 'column',
        justifyContent: 'space-between',
        backgroundColor: '#0b1323',
        padding: '68px 80px',
        fontFamily: 'sans-serif',
      },
    },
      // Top bar
      h('div', { style: { display: 'flex', alignItems: 'center', gap: 16 } },
        h('div', { style: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#adc6ff' } }),
        h('div', { style: { color: '#adc6ff', fontSize: 18, fontWeight: 300, letterSpacing: 3, textTransform: 'uppercase' } },
          'bpr.thestardrive.com'
        ),
      ),

      // Title block
      h('div', { style: { display: 'flex', flexDirection: 'column', gap: 20 } },
        h('div', { style: { color: '#dbe2f8', fontSize: 72, fontWeight: 300, lineHeight: 1.05, letterSpacing: -2 } },
          title
        ),
        h('div', { style: { color: '#8c909f', fontSize: 26, fontWeight: 300 } },
          sub
        ),
      ),

      // Stats row
      h('div', { style: { display: 'flex', gap: 56, alignItems: 'flex-end' } },
        ...[ ['87', 'predictions'], ['Zero', 'free parameters'], ['1,225', 'tests passing'], ['MIT', 'licensed'] ]
          .map(([num, label]) =>
            h('div', { key: label, style: { display: 'flex', flexDirection: 'column', gap: 6 } },
              h('div', { style: { color: '#adc6ff', fontSize: 34, fontWeight: 300, lineHeight: 1 } }, num),
              h('div', { style: { color: '#424754', fontSize: 12, textTransform: 'uppercase', letterSpacing: 2 } }, label),
            )
          )
      ),
    ),
    { width: 1200, height: 630 }
  );
}
