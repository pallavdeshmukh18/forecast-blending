/**
 * Centralized HTTP API Client for BharatWeather Frontend
 * Connects React UI to the FastAPI operational backend.
 * Uses VITE_API_BASE_URL environment variable with fallback to http://localhost:8000.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = 25000;

export class ApiError extends Error {
  constructor({ message, status, endpoint, details = null, isNetworkError = false }) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.endpoint = endpoint;
    this.details = details;
    this.isNetworkError = isNetworkError;
    this.timestamp = new Date().toISOString();
  }
}

/**
 * Perform an HTTP request to the FastAPI backend
 */
async function request(endpoint, options = {}) {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const timeoutMs = options.timeout ?? DEFAULT_TIMEOUT_MS;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = {
    Accept: 'application/json',
    ...(options.headers || {}),
  };

  // Only set Content-Type for requests with a payload to allow simple CORS GET requests without preflight
  if (options.body && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const fetchOptions = {
    ...options,
    signal: options.signal || controller.signal,
    headers,
  };

  if (fetchOptions.body && typeof fetchOptions.body === 'object' && !(fetchOptions.body instanceof FormData)) {
    fetchOptions.body = JSON.stringify(fetchOptions.body);
  }

  let url = `${BASE_URL}${cleanEndpoint}`;

  try {
    let res;
    try {
      res = await fetch(url, fetchOptions);
    } catch (initialErr) {
      // If localhost failed (e.g. IPv6 issue on Windows), transparently retry with 127.0.0.1
      if (url.includes('localhost:8000')) {
        const fallbackUrl = url.replace('localhost:8000', '127.0.0.1:8000');
        res = await fetch(fallbackUrl, fetchOptions);
        url = fallbackUrl;
      } else if (url.includes('127.0.0.1:8000')) {
        const fallbackUrl = url.replace('127.0.0.1:8000', 'localhost:8000');
        res = await fetch(fallbackUrl, fetchOptions);
        url = fallbackUrl;
      } else {
        throw initialErr;
      }
    }

    clearTimeout(timeoutId);

    let parsedJson = null;
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try {
        parsedJson = await res.json();
      } catch (parseErr) {
        parsedJson = null;
      }
    } else {
      const text = await res.text();
      parsedJson = text ? { message: text } : null;
    }

    if (!res.ok) {
      const errMsg = parsedJson?.detail || parsedJson?.message || `HTTP ${res.status}: ${res.statusText}`;
      throw new ApiError({
        message: errMsg,
        status: res.status,
        endpoint,
        details: parsedJson,
        isNetworkError: false,
      });
    }

    return parsedJson;
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof ApiError) {
      throw err;
    }

    const isAbort = err.name === 'AbortError';
    const netErrMsg = isAbort
      ? `Request timed out after ${timeoutMs}ms for ${endpoint}`
      : `Network connection failed to ${url} (${err.message})`;

    console.warn(`[BharatWeather API] ${netErrMsg}`);

    throw new ApiError({
      message: netErrMsg,
      status: 0,
      endpoint,
      details: err,
      isNetworkError: true,
    });
  }
}

/**
 * Safe request wrapper that returns { data, error, isLive, status } without throwing
 */
async function safeRequest(endpoint, options = {}, fallback = null) {
  try {
    const data = await request(endpoint, options);
    return { data, error: null, isLive: true, status: 200 };
  } catch (err) {
    return {
      data: fallback,
      error: err,
      isLive: false,
      status: err.status || 0,
    };
  }
}

export const apiClient = {
  baseUrl: BASE_URL,
  request,
  safeRequest,
  get: (endpoint, options = {}) => request(endpoint, { ...options, method: 'GET' }),
  post: (endpoint, body, options = {}) => request(endpoint, { ...options, method: 'POST', body }),
  safeGet: (endpoint, options = {}, fallback = null) =>
    safeRequest(endpoint, { ...options, method: 'GET' }, fallback),
  safePost: (endpoint, body, options = {}, fallback = null) =>
    safeRequest(endpoint, { ...options, method: 'POST', body }, fallback),
};

export default apiClient;
