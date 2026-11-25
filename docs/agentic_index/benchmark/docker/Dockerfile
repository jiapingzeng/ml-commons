#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# OpenSearch with the ml-commons plugin built from this branch. The bundled skills plugin depends on ml-commons and is
# removed first, as in ml-commons CI. Build context: this directory, with the plugin zip copied next to this file.
ARG OPENSEARCH_IMAGE=opensearchstaging/opensearch:3.10.0
FROM ${OPENSEARCH_IMAGE}
ARG PLUGIN_ZIP=opensearch-ml-3.10.0.0-SNAPSHOT.zip
RUN /usr/share/opensearch/bin/opensearch-plugin remove opensearch-skills \
 && /usr/share/opensearch/bin/opensearch-plugin remove opensearch-ml
COPY ${PLUGIN_ZIP} /tmp/opensearch-ml.zip
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:/tmp/opensearch-ml.zip
