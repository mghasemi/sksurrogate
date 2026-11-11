/**
 * Login screen shown by the shell when the API enforces authentication and no
 * session is active.
 *
 * The form posts to ``/api/auth/login``; on success the server sets the
 * ``HttpOnly`` session cookie and the provider refreshes, at which point the
 * shell renders the application. Failures are throttled server-side (5 per
 * username+client per 15 minutes) and answered with an indistinguishable 401,
 * so the UI never hints at which half of the credentials was wrong.
 */

import { useState } from "react";

import { useAuth } from "../lib/auth-context";

export default function LoginPage() {
  const { login, status } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username.trim(), password);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(
        message.includes("429")
          ? "Too many failed attempts — wait a few minutes before trying again."
          : "Invalid username or password.",
      );
    } finally {
      setBusy(false);
    }
  };

  // With no accounts yet there is nothing to log in with: show the exact
  // server-side command instead of a form that cannot succeed.
  const bootstrap = status?.bootstrap_required ?? false;

  return (
    <div className="shell" style={{ display: "block" }}>
      <div
        className="card"
        style={{ maxWidth: 460, margin: "12vh auto 0", padding: 24 }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <div className="logo">SK</div>
          <div>
            <div className="name" style={{ fontWeight: 600 }}>SKSurrogate</div>
            <div className="sub muted" style={{ fontSize: 12 }}>Control Plane</div>
          </div>
        </div>

        <h2 style={{ marginTop: 0 }}>Sign in</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          This deployment requires an account. Sessions are kept in an HttpOnly cookie and can
          be revoked by an administrator.
        </p>

        {bootstrap ? (
          <div className="warn-note">
            <p style={{ marginTop: 0 }}>
              <strong>No accounts exist yet.</strong> Create the first administrator on the server:
            </p>
            <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
              python -m api.auth_cli create-admin --username you
            </pre>
            <p style={{ marginBottom: 0 }}>
              or start the API with <code>SKSURROGATE_ADMIN_USERNAME</code> and{" "}
              <code>SKSURROGATE_ADMIN_PASSWORD</code> set once, then reload this page.
            </p>
          </div>
        ) : (
          <form onSubmit={submit}>
            <div className="field" style={{ marginBottom: 12 }}>
              <label htmlFor="login-username">Username</label>
              <input
                id="login-username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
                required
              />
            </div>
            <div className="field" style={{ marginBottom: 16 }}>
              <label htmlFor="login-password">Password</label>
              <input
                id="login-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                required
              />
            </div>
            {error && <p className="error-note">{error}</p>}
            <button className="btn primary" type="submit" disabled={busy || !username || !password}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
