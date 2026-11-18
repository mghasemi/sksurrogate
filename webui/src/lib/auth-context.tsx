/**
 * Authentication state for the control-plane UI.
 *
 * The browser never holds a token: the API session lives in an ``HttpOnly``
 * cookie, so "am I signed in?" is answered by asking the server
 * (``/api/auth/me``). Two facts decide the UI's shape:
 *
 * * ``auth_enforced`` from ``/api/auth/status`` — false means the API is open
 *   (local single-user use) and every request runs as the anonymous principal,
 *   so no login screen is shown and nothing is filtered;
 * * the resolved ``Principal`` — when enforcement is on and the principal is
 *   anonymous, the shell renders the login page instead of the app.
 */

import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  ApiError,
  authStatus,
  currentPrincipal,
  login as apiLogin,
  logout as apiLogout,
} from "../api/client";
import type { AuthStatus, Principal } from "../api/client";

interface AuthContextValue {
  status: AuthStatus | undefined;
  me: Principal | undefined;
  /** True while the initial status/identity probe is in flight. */
  loading: boolean;
  /** True when the server enforces auth and nobody is signed in. */
  needsLogin: boolean;
  isAdmin: boolean;
  error: unknown;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isUnauthorised(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();

  const statusQ = useQuery({
    queryKey: ["auth-status"],
    queryFn: authStatus,
    retry: false,
    staleTime: 30_000,
  });

  const enforced = statusQ.data?.auth_enforced ?? false;
  const meQ = useQuery({
    queryKey: ["auth-me"],
    queryFn: currentPrincipal,
    retry: false,
    staleTime: 30_000,
    enabled: !!statusQ.data,
  });

  const login = useCallback(
    async (username: string, password: string) => {
      await apiLogin(username, password);
      await qc.invalidateQueries();
    },
    [qc],
  );

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } finally {
      qc.clear();
      await qc.invalidateQueries();
    }
  }, [qc]);

  const refresh = useCallback(async () => {
    await qc.invalidateQueries({ queryKey: ["auth-status"] });
    await qc.invalidateQueries({ queryKey: ["auth-me"] });
  }, [qc]);

  const me = meQ.data;
  const needsLogin = enforced && (!me || me.source === "anonymous" || isUnauthorised(meQ.error));

  const value = useMemo<AuthContextValue>(
    () => ({
      status: statusQ.data,
      me,
      loading: statusQ.isLoading || (enforced && meQ.isLoading),
      needsLogin,
      isAdmin: !!me?.is_admin,
      error: statusQ.error ?? (isUnauthorised(meQ.error) ? null : meQ.error),
      login,
      logout,
      refresh,
    }),
    [statusQ.data, statusQ.isLoading, statusQ.error, me, meQ.isLoading, meQ.error, enforced, needsLogin, login, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}

/** Convenience for pages: the signed-in username, or a label for anonymous access. */
export function useUsername(): string {
  const { me } = useAuth();
  if (!me) return "";
  if (me.user) return me.user.username;
  return me.source === "static" ? "shared key" : me.username;
}
