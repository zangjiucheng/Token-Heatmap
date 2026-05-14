/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class SchemaService {
    /**
     * Get Schema
     * Return ``docs/web/trace.schema.json`` byte-for-byte.
     *
     * Uses ``application/schema+json`` as the content type so caches and the
     * frontend can distinguish a schema document from a plain JSON payload.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getSchemaSchemaGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/schema',
        });
    }
    /**
     * Get Activation Schema
     * Return ``docs/web/activation.schema.json`` byte-for-byte.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getActivationSchemaSchemaActivationGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/schema/activation',
        });
    }
    /**
     * Get Activation Diff Schema
     * Return ``docs/web/activation-diff.schema.json`` byte-for-byte.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getActivationDiffSchemaSchemaActivationDiffGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/schema/activation-diff',
        });
    }
    /**
     * Get Activation Sidecar Schema
     * Return ``docs/web/activation-sidecar.schema.json`` byte-for-byte.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getActivationSidecarSchemaSchemaActivationSidecarGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/schema/activation-sidecar',
        });
    }
}
