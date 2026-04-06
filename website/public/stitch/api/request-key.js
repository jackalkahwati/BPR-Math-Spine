const crypto = require('crypto');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { name, email, institution, purpose } = req.body || {};

  if (!name || !email || !purpose) {
    return res.status(400).json({ error: 'Name, email, and research purpose are required.' });
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ error: 'Invalid email address.' });
  }

  const key = 'bpr_' + crypto.randomBytes(24).toString('hex');
  const resendKey = process.env.RESEND_API_KEY;

  if (!resendKey) {
    return res.status(500).json({ error: 'Email service not configured.' });
  }

  const send = (to, subject, html) =>
    fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${resendKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ from: 'BPR Research <research@bpr.thestardrive.com>', to, subject, html }),
    });

  // Email to researcher
  await send(email, 'Your BPR API Key', `
    <div style="font-family:monospace;background:#0b1323;color:#dbe2f8;padding:32px;max-width:580px;border-radius:12px">
      <h2 style="color:#adc6ff;font-weight:300;margin-top:0">BPR API Access Granted</h2>
      <p>Hello ${escapeHtml(name)},</p>
      <p>Your API key for the Boundary Phase Resonance research API:</p>
      <div style="background:#1a2335;border:1px solid #2d4a7a;padding:16px;border-radius:8px;margin:24px 0;word-break:break-all">
        <code style="color:#adc6ff;font-size:14px">${key}</code>
      </div>
      <p style="color:#8c909f;font-size:13px">Store this securely — we will not show it again.</p>
      <p style="color:#8c909f;font-size:13px">Include it in API requests as a header:<br>
        <code style="color:#adc6ff">X-BPR-Key: ${key}</code>
      </p>
      <p>Documentation: <a href="https://bpr.thestardrive.com/docs" style="color:#adc6ff">bpr.thestardrive.com/docs</a></p>
      <hr style="border:none;border-top:1px solid #2d3546;margin:24px 0">
      <p style="color:#4a5568;font-size:11px">Boundary Phase Resonance &middot; StarDrive Inc. &middot; jack@thestardrive.com</p>
    </div>
  `);

  // Admin notification
  await send('jack@thestardrive.com', `New BPR API Key — ${name}`, `
    <p><strong>Name:</strong> ${escapeHtml(name)}</p>
    <p><strong>Email:</strong> ${escapeHtml(email)}</p>
    <p><strong>Institution:</strong> ${escapeHtml(institution || 'Not specified')}</p>
    <p><strong>Purpose:</strong> ${escapeHtml(purpose)}</p>
    <p><strong>Key:</strong> <code>${key}</code></p>
    <p><strong>Time:</strong> ${new Date().toISOString()}</p>
  `);

  return res.status(200).json({ key });
};

function escapeHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
