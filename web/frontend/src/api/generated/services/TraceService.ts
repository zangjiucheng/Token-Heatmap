/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_convert_csv_trace_convert_csv_post } from '../models/Body_convert_csv_trace_convert_csv_post';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class TraceService {
    /**
     * Convert Csv
     * Accept the CSV emitted by ``trace_to_dataframe`` and return JSON.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static convertCsvTraceConvertCsvPost({
        formData,
    }: {
        formData: Body_convert_csv_trace_convert_csv_post,
    }): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/trace/convert-csv',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
