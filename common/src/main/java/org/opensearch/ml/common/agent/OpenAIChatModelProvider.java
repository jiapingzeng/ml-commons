/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.agent;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.text.StringEscapeUtils;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLAgentType;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.connector.ConnectorAction;
import org.opensearch.ml.common.connector.ConnectorClientConfig;
import org.opensearch.ml.common.connector.HttpConnector;
import org.opensearch.ml.common.input.execute.agent.ContentBlock;
import org.opensearch.ml.common.input.execute.agent.ImageContent;
import org.opensearch.ml.common.input.execute.agent.Message;
import org.opensearch.ml.common.input.execute.agent.SourceType;
import org.opensearch.ml.common.model.ModelProvider;
import org.opensearch.ml.common.transport.register.MLRegisterModelInput;

import lombok.extern.log4j.Log4j2;

/**
 * Model provider for OpenAI Chat Completion API.
 *
 * This provider uses template-based parameter substitution to create request bodies
 * for the OpenAI Chat Completion API. It supports text, multimodal content (images),
 * and message-based conversations.
 *
 * OpenAI API Format:
 * - System messages are included in the messages array with role "system"
 * - Messages array format: [{"role": "user|assistant|system", "content": "..."}]
 * - Multimodal content uses: {"type": "text|image_url", ...}
 * - Images: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
 * - Only BASE64 image sources are supported
 *
 * Template parameters use ${parameters.} prefix for uniformity with Bedrock provider.
 */
@Log4j2
public class OpenAIChatModelProvider extends ModelProvider {

    private static final String DEFAULT_ENDPOINT = "api.openai.com";

    private static final String REQUEST_BODY_TEMPLATE = "{\"model\":\"${parameters.model}\",\"messages\":${parameters.messages}}";

    // Body templates for different input types
    private static final String TEXT_INPUT_BODY_TEMPLATE = "[{\"role\":\"user\",\"content\":\"${parameters.user_text}\"}]";

    private static final String CONTENT_BLOCKS_BODY_TEMPLATE = "[{\"role\":\"user\",\"content\":[${parameters.content_array}]}]";

    // Content block templates for multi-modal content
    private static final String TEXT_CONTENT_TEMPLATE = "{\"type\":\"text\",\"text\":\"${parameters.content_text}\"}";

    private static final String IMAGE_CONTENT_TEMPLATE = "{\"type\":\"image_url\",\"image_url\":{\"url\":\"${parameters.image_url}\"}}";

    private static final String MESSAGE_TEMPLATE = "{\"role\":\"${parameters.msg_role}\",\"content\":${parameters.msg_content}}";

    @Override
    public Connector createConnector(String modelId, Map<String, String> credential, Map<String, String> modelParameters) {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("endpoint", DEFAULT_ENDPOINT);
        parameters.put("model", modelId);

        // Override with any provided model parameters
        if (modelParameters != null) {
            parameters.putAll(modelParameters);
        }

        Map<String, String> headers = new HashMap<>();
        headers.put("Content-Type", "application/json");
        headers.put("Authorization", "Bearer ${credential.openAI_key}");

        ConnectorAction predictAction = ConnectorAction
            .builder()
            .actionType(ConnectorAction.ActionType.PREDICT)
            .method("POST")
            .url("https://${parameters.endpoint}/v1/chat/completions")
            .headers(headers)
            .requestBody(REQUEST_BODY_TEMPLATE)
            .build();

        // Set agent connector to have default 3 retries
        ConnectorClientConfig connectorClientConfig = new ConnectorClientConfig();
        connectorClientConfig.setMaxRetryTimes(3);

        return HttpConnector
            .builder()
            .name("Auto-generated OpenAI connector for Agent")
            .description("Auto-generated connector for OpenAI Chat Completion API")
            .version("1")
            .protocol("http")
            .parameters(parameters)
            .credential(credential != null ? credential : new HashMap<>())
            .actions(List.of(predictAction))
            .connectorClientConfig(connectorClientConfig)
            .build();
    }

    @Override
    public MLRegisterModelInput createModelInput(String modelName, Connector connector, Map<String, String> modelParameters) {
        return MLRegisterModelInput
            .builder()
            .functionName(FunctionName.REMOTE)
            .modelName("Auto-generated model for " + modelName)
            .description("Auto-generated model for agent")
            .connector(connector)
            .build();
    }

    @Override
    public String getLLMInterface() {
        return "openai/v1/chat/completions";
    }

    @Override
    public Map<String, String> mapTextInput(String text, MLAgentType type) {
        Map<String, String> parameters = new HashMap<>();

        // Use StringSubstitutor for parameter replacement
        Map<String, String> templateParams = new HashMap<>();
        templateParams.put("user_text", StringEscapeUtils.escapeJson(text));

        StringSubstitutor substitutor = new StringSubstitutor(templateParams, "${parameters.", "}");
        String messages = substitutor.replace(TEXT_INPUT_BODY_TEMPLATE);
        parameters.put("messages", messages);

        return parameters;
    }

    @Override
    public Map<String, String> mapContentBlocks(List<ContentBlock> contentBlocks, MLAgentType type) {
        Map<String, String> parameters = new HashMap<>();

        // Build content array from blocks
        String contentArray = buildContentArrayFromBlocks(contentBlocks);
        Map<String, String> templateParams = new HashMap<>();
        templateParams.put("content_array", contentArray);

        StringSubstitutor substitutor = new StringSubstitutor(templateParams, "${parameters.", "}");
        String messages = substitutor.replace(CONTENT_BLOCKS_BODY_TEMPLATE);
        parameters.put("messages", messages);

        return parameters;
    }

    @Override
    public Map<String, String> mapMessages(List<Message> messages, MLAgentType type) {
        Map<String, String> parameters = new HashMap<>();
        String messagesString = buildMessagesArray(messages);
        parameters.put("messages", messagesString);
        return parameters;
    }

    /**
     * Builds content array from content blocks using templates for OpenAI Chat Completion API.
     * Supports text and image content types. Only BASE64 image sources are supported.
     * Documents and videos are not supported by OpenAI and will be logged as warnings.
     */
    private String buildContentArrayFromBlocks(List<ContentBlock> blocks) {
        if (blocks == null || blocks.isEmpty()) {
            return "";
        }

        StringBuilder contentArray = new StringBuilder();
        boolean first = true;
        for (ContentBlock block : blocks) {
            switch (block.getType()) {
                case TEXT:
                    if (!first) {
                        contentArray.append(",");
                    }
                    first = false;

                    Map<String, Object> textParams = new HashMap<>();
                    textParams.put("content_text", StringEscapeUtils.escapeJson(block.getText()));
                    StringSubstitutor textSubstitutor = new StringSubstitutor(textParams, "${parameters.", "}");
                    contentArray.append(textSubstitutor.replace(TEXT_CONTENT_TEMPLATE));
                    break;

                case IMAGE:
                    ImageContent image = block.getImage();
                    
                    // Only support BASE64 source type
                    if (image.getType() != SourceType.BASE64) {
                        throw new IllegalArgumentException(
                            "Only BASE64 image sources are supported for OpenAI. URL-based images are not supported."
                        );
                    }

                    if (!first) {
                        contentArray.append(",");
                    }
                    first = false;

                    Map<String, Object> imageParams = new HashMap<>();
                    String imageUrl = formatImageDataUrl(image);
                    imageParams.put("image_url", StringEscapeUtils.escapeJson(imageUrl));
                    StringSubstitutor imageSubstitutor = new StringSubstitutor(imageParams, "${parameters.", "}");
                    contentArray.append(imageSubstitutor.replace(IMAGE_CONTENT_TEMPLATE));
                    break;

                case DOCUMENT:
                    log.warn("Document content is not supported by OpenAI Chat Completion API. Skipping document block.");
                    break;

                case VIDEO:
                    log.warn("Video content is not supported by OpenAI Chat Completion API. Skipping video block.");
                    break;

                default:
                    log.warn("Unsupported content type: {}. Skipping block.", block.getType());
                    break;
            }
        }

        return contentArray.toString();
    }

    /**
     * Builds messages array using templates for OpenAI Chat Completion API.
     * Converts messages to OpenAI format where each message has role and content.
     */
    private String buildMessagesArray(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "[]";
        }

        StringBuilder messagesArray = new StringBuilder();
        messagesArray.append("[");
        boolean first = true;

        for (Message message : messages) {
            if (!first) {
                messagesArray.append(",");
            }
            first = false;

            String contentString = buildMessageContent(message);
            Map<String, Object> msgParams = new HashMap<>();
            msgParams.put("msg_role", message.getRole());
            msgParams.put("msg_content", contentString);
            StringSubstitutor msgSubstitutor = new StringSubstitutor(msgParams, "${parameters.", "}");
            messagesArray.append(msgSubstitutor.replace(MESSAGE_TEMPLATE));
        }

        messagesArray.append("]");
        return messagesArray.toString();
    }

    /**
     * Builds content for a single message. For simple text messages, returns a string.
     * For multimodal messages, returns an array of content blocks.
     */
    private String buildMessageContent(Message message) {
        if (message.getContent() == null || message.getContent().isEmpty()) {
            return "\"\"";
        }

        // Check if this is a simple text-only message
        if (message.getContent().size() == 1 && message.getContent().get(0).getType() == org.opensearch.ml.common.input.execute.agent.ContentType.TEXT) {
            String text = message.getContent().get(0).getText();
            return "\"" + StringEscapeUtils.escapeJson(text) + "\"";
        }

        // Multi-modal content - build array of content blocks
        String contentArray = buildContentArrayFromBlocks(message.getContent());
        return "[" + contentArray + "]";
    }

    /**
     * Formats image data as a data URL for OpenAI API.
     * Creates a data URL with the format: data:image/{format};base64,{data}
     *
     * @param image the image content (must be BASE64 type)
     * @return formatted data URL
     */
    private String formatImageDataUrl(ImageContent image) {
        String format = image.getFormat() != null ? image.getFormat() : "jpeg";
        return "data:image/" + format + ";base64," + image.getData();
    }
}
