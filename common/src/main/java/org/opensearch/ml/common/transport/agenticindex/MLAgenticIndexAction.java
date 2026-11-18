/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.agenticindex;

import org.opensearch.action.ActionType;

public class MLAgenticIndexAction extends ActionType<MLAgenticIndexResponse> {
    public static final MLAgenticIndexAction INSTANCE = new MLAgenticIndexAction();
    public static final String NAME = "cluster:admin/opensearch/ml/agentic_index/query";

    private MLAgenticIndexAction() {
        super(NAME, MLAgenticIndexResponse::new);
    }
}
