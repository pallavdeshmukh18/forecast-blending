/**
 * Centralized API Layer for SkyCast
 * Re-exports all API clients, endpoint functions, and data adapters.
 */

export { apiClient, ApiError } from './client';
export * from './health';
export * from './metadata';
export * from './models';
export * from './forecast';
export * as overviewAdapter from './adapters/overviewAdapter';
