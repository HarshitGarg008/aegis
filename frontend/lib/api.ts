const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

type Credentials = {
  email: string;
  password: string;
};

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  const data = (await response.json()) as T;

  if (!response.ok) {
    throw new Error(
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: unknown }).error)
        : "Request failed"
    );
  }

  return data;
}

export async function login(credentials: Credentials) {
  return request<{ idToken: string; email: string; uid: string }>("/login", {
    method: "POST",
    body: JSON.stringify(credentials)
  });
}

export async function signup(credentials: Credentials) {
  return request<{ uid: string; email: string }>("/signup", {
    method: "POST",
    body: JSON.stringify(credentials)
  });
}
