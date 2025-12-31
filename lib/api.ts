// API utility with retry logic and error handling

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5328';
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;

interface ApiOptions extends RequestInit {
    timeout?: number;
    retries?: number;
    retryDelay?: number;
}

class ApiError extends Error {
    constructor(
        message: string,
        public status?: number,
        public data?: any
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Fetch with timeout support
 */
async function fetchWithTimeout(
    url: string,
    options: RequestInit = {},
    timeout: number = DEFAULT_TIMEOUT
): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
            throw new ApiError('Request timeout', 408);
        }
        throw error;
    }
}

/**
 * Main API call function with retry logic
 */
export async function apiCall<T = any>(
    endpoint: string,
    options: ApiOptions = {}
): Promise<T> {
    const {
        timeout = DEFAULT_TIMEOUT,
        retries = MAX_RETRIES,
        retryDelay = 1000,
        ...fetchOptions
    } = options;

    const url = endpoint.startsWith('http')
        ? endpoint
        : `${API_BASE_URL}${endpoint}`;

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < retries; attempt++) {
        try {
            const response = await fetchWithTimeout(url, fetchOptions, timeout);

            // Handle non-OK responses
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new ApiError(
                    errorData.error || `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorData
                );
            }

            // Parse JSON response
            const data = await response.json();
            return data as T;

        } catch (error) {
            lastError = error as Error;

            // Don't retry on client errors (4xx) except 408 (timeout) and 429 (rate limit)
            if (error instanceof ApiError) {
                const status = error.status;
                if (status && status >= 400 && status < 500 && status !== 408 && status !== 429) {
                    throw error;
                }
            }

            // If this is the last attempt, throw the error
            if (attempt === retries - 1) {
                throw error;
            }

            // Wait before retrying (exponential backoff)
            const delay = retryDelay * Math.pow(2, attempt);
            console.warn(`API call failed (attempt ${attempt + 1}/${retries}), retrying in ${delay}ms...`, error);
            await sleep(delay);
        }
    }

    // This should never be reached, but TypeScript needs it
    throw lastError || new ApiError('Unknown error occurred');
}

/**
 * Convenience methods for common HTTP verbs
 */
export const api = {
    get: <T = any>(endpoint: string, options?: ApiOptions) =>
        apiCall<T>(endpoint, { ...options, method: 'GET' }),

    post: <T = any>(endpoint: string, data?: any, options?: ApiOptions) =>
        apiCall<T>(endpoint, {
            ...options,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            body: data ? JSON.stringify(data) : undefined,
        }),

    put: <T = any>(endpoint: string, data?: any, options?: ApiOptions) =>
        apiCall<T>(endpoint, {
            ...options,
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            body: data ? JSON.stringify(data) : undefined,
        }),

    delete: <T = any>(endpoint: string, options?: ApiOptions) =>
        apiCall<T>(endpoint, { ...options, method: 'DELETE' }),

    // Special method for file uploads
    upload: async <T = any>(
        endpoint: string,
        formData: FormData,
        options?: ApiOptions
    ) => {
        return apiCall<T>(endpoint, {
            ...options,
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - browser will set it with boundary
        });
    },
};

/**
 * Hook for handling API errors in React components
 */
export function useApiError() {
    const handleError = (error: unknown): string => {
        if (error instanceof ApiError) {
            return error.message;
        }
        if (error instanceof Error) {
            return error.message;
        }
        return 'An unexpected error occurred';
    };

    return { handleError };
}

export { ApiError };
