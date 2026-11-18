/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.agenticindex;

import java.io.IOException;
import java.util.Map;

import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import lombok.Getter;

/**
 * Response of the agentic index APIs. The result is a plain map (answer, citations, passages, trace)
 * because the POC response shape is still evolving.
 */
@Getter
public class MLAgenticIndexResponse extends ActionResponse implements ToXContentObject {

    private final Map<String, Object> result;

    public MLAgenticIndexResponse(Map<String, Object> result) {
        this.result = result;
    }

    public MLAgenticIndexResponse(StreamInput in) throws IOException {
        super(in);
        this.result = in.readMap();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeMap(result);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        return builder.map(result);
    }
}
