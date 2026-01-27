/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.agent;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.ml.common.MLAgentType;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.connector.HttpConnector;
import org.opensearch.ml.common.input.execute.agent.AgentInput;
import org.opensearch.ml.common.input.execute.agent.ContentBlock;
import org.opensearch.ml.common.input.execute.agent.ContentType;
import org.opensearch.ml.common.input.execute.agent.DocumentContent;
import org.opensearch.ml.common.input.execute.agent.ImageContent;
import org.opensearch.ml.common.input.execute.agent.Message;
import org.opensearch.ml.common.input.execute.agent.SourceType;
import org.opensearch.ml.common.input.execute.agent.VideoContent;

public class OpenAIChatModelProviderTest {

    private OpenAIChatModelProvider provider;

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Before
    public void setUp() {
        provider = new OpenAIChatModelProvider();
    }

    @Test
    public void testGetLLMInterface() {
        // Act
        String result = provider.getLLMInterface();

        // Assert
        assertEquals("openai/v1/chat/completions", result);
    }

    @Test
    public void testCreateConnector_WithFullParameters() {
        // Arrange
        String modelId = "gpt-4o";
        Map<String, String> credential = new HashMap<>();
        credential.put("openAI_key", "sk-test-key-123");

        Map<String, String> modelParameters = new HashMap<>();
        modelParameters.put("temperature", "0.7");
        modelParameters.put("max_tokens", "1000");

        // Act
        Connector connector = provider.createConnector(modelId, credential, modelParameters);

        // Assert
        assertNotNull(connector);
        assertTrue(connector instanceof HttpConnector);
        HttpConnector httpConnector = (HttpConnector) connector;
        assertEquals("Auto-generated OpenAI connector for Agent", httpConnector.getName());
        assertEquals("Auto-generated connector for OpenAI Chat Completion API", httpConnector.getDescription());
        assertEquals("http", httpConnector.getProtocol());
        assertEquals(modelId, httpConnector.getParameters().get("model"));
        assertEquals("api.openai.com", httpConnector.getParameters().get("endpoint"));
        assertEquals("0.7", httpConnector.getParameters().get("temperature"));
        assertEquals("1000", httpConnector.getParameters().get("max_tokens"));
        assertNotNull(httpConnector.getActions());
        assertEquals(1, httpConnector.getActions().size());
    }

    @Test
    public void testCreateConnector_WithDefaultEndpoint() {
        // Arrange
        String modelId = "gpt-3.5-turbo";
        Map<String, String> credential = new HashMap<>();
        credential.put("openAI_key", "sk-test-key");

        // Act
        Connector connector = provider.createConnector(modelId, credential, null);

        // Assert
        assertNotNull(connector);
        assertTrue(connector instanceof HttpConnector);
        HttpConnector httpConnector = (HttpConnector) connector;
        assertEquals("api.openai.com", httpConnector.getParameters().get("endpoint"));
    }

    @Test
    public void testCreateConnector_WithNullCredential() {
        // Arrange
        String modelId = "gpt-4";

        // Act
        Connector connector = provider.createConnector(modelId, null, null);

        // Assert
        assertNotNull(connector);
        assertTrue(connector instanceof HttpConnector);
    }

    @Test
    public void testCreateConnector_VerifyRetryConfiguration() {
        // Arrange
        String modelId = "gpt-4";
        Map<String, String> credential = new HashMap<>();
        credential.put("openAI_key", "sk-test");

        // Act
        Connector connector = provider.createConnector(modelId, credential, null);

        // Assert
        assertTrue(connector instanceof HttpConnector);
        HttpConnector httpConnector = (HttpConnector) connector;
        assertNotNull(httpConnector.getConnectorClientConfig());
        assertEquals(Integer.valueOf(3), httpConnector.getConnectorClientConfig().getMaxRetryTimes());
    }

    @Test
    public void testMapTextInput_SimpleText() {
        // Arrange
        String text = "Hello, how are you?";

        // Act
        Map<String, String> result = provider.mapTextInput(text, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"role\":\"user\""));
        assertTrue(messages.contains("\"content\":\"Hello, how are you?\""));
        assertTrue(messages.startsWith("[") && messages.endsWith("]"));
    }

    @Test
    public void testMapTextInput_WithSpecialCharacters() {
        // Arrange
        String text = "Text with \"quotes\" and \n newlines";

        // Act
        Map<String, String> result = provider.mapTextInput(text, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\\\""));
        assertTrue(messages.contains("\\n"));
    }

    @Test
    public void testMapContentBlocks_TextOnly() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Hello world");
        blocks.add(textBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"type\":\"text\""));
        assertTrue(messages.contains("\"text\":\"Hello world\""));
    }

    @Test
    public void testMapContentBlocks_ImageWithBase64() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, "png", "base64encodeddata");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"type\":\"image_url\""));
        assertTrue(messages.contains("data:image/png;base64,"));
        assertTrue(messages.contains("base64encodeddata"));
    }

    @Test
    public void testMapContentBlocks_ImageWithURLSourceType_ThrowsException() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.URL, "jpeg", "https://example.com/image.jpg");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act & Assert
        try {
            provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);
            fail("Should throw IllegalArgumentException for URL image source");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Only BASE64 image sources are supported for OpenAI"));
        }
    }

    @Test
    public void testMapContentBlocks_DocumentNotSupported() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock docBlock = new ContentBlock();
        docBlock.setType(ContentType.DOCUMENT);
        DocumentContent doc = new DocumentContent(SourceType.BASE64, "pdf", "base64pdfdata");
        docBlock.setDocument(doc);
        blocks.add(docBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert - document should be skipped with warning, not throw exception
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
    }

    @Test
    public void testMapContentBlocks_VideoNotSupported() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock videoBlock = new ContentBlock();
        videoBlock.setType(ContentType.VIDEO);
        VideoContent video = new VideoContent(SourceType.BASE64, "mp4", "base64videodata");
        videoBlock.setVideo(video);
        blocks.add(videoBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert - video should be skipped with warning, not throw exception
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
    }

    @Test
    public void testMapContentBlocks_MultipleBlocks() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();

        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Describe this image:");
        blocks.add(textBlock);

        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, "png", "imagedata");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"text\":\"Describe this image:\""));
        assertTrue(messages.contains("\"type\":\"image_url\""));
    }

    @Test
    public void testMapContentBlocks_EmptyList() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
    }

    @Test
    public void testMapContentBlocks_NullList() {
        // Act
        Map<String, String> result = provider.mapContentBlocks(null, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
    }

    @Test
    public void testMapMessages_SingleTextMessage() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        List<ContentBlock> content = new ArrayList<>();
        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Hello");
        content.add(textBlock);

        Message message = new Message("user", content);
        messages.add(message);

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"role\":\"user\""));
        assertTrue(messagesStr.contains("\"Hello\""));
        assertTrue(messagesStr.startsWith("[") && messagesStr.endsWith("]"));
    }

    @Test
    public void testMapMessages_MultipleMessages() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        List<ContentBlock> userContent = new ArrayList<>();
        ContentBlock userBlock = new ContentBlock();
        userBlock.setType(ContentType.TEXT);
        userBlock.setText("Hello");
        userContent.add(userBlock);
        messages.add(new Message("user", userContent));

        List<ContentBlock> assistantContent = new ArrayList<>();
        ContentBlock assistantBlock = new ContentBlock();
        assistantBlock.setType(ContentType.TEXT);
        assistantBlock.setText("Hi there!");
        assistantContent.add(assistantBlock);
        messages.add(new Message("assistant", assistantContent));

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"role\":\"user\""));
        assertTrue(messagesStr.contains("\"role\":\"assistant\""));
        assertTrue(messagesStr.contains("Hello"));
        assertTrue(messagesStr.contains("Hi there!"));
    }

    @Test
    public void testMapMessages_WithSystemMessage() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        List<ContentBlock> systemContent = new ArrayList<>();
        ContentBlock systemBlock = new ContentBlock();
        systemBlock.setType(ContentType.TEXT);
        systemBlock.setText("You are a helpful assistant.");
        systemContent.add(systemBlock);
        messages.add(new Message("system", systemContent));

        List<ContentBlock> userContent = new ArrayList<>();
        ContentBlock userBlock = new ContentBlock();
        userBlock.setType(ContentType.TEXT);
        userBlock.setText("Hello");
        userContent.add(userBlock);
        messages.add(new Message("user", userContent));

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"role\":\"system\""));
        assertTrue(messagesStr.contains("\"role\":\"user\""));
        assertTrue(messagesStr.contains("You are a helpful assistant."));
    }

    @Test
    public void testMapMessages_MultimodalMessage() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        List<ContentBlock> content = new ArrayList<>();
        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("What's in this image?");
        content.add(textBlock);

        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, "jpeg", "base64data");
        imageBlock.setImage(image);
        content.add(imageBlock);

        Message message = new Message("user", content);
        messages.add(message);

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"type\":\"text\""));
        assertTrue(messagesStr.contains("\"type\":\"image_url\""));
        assertTrue(messagesStr.contains("data:image/jpeg;base64,"));
    }

    @Test
    public void testMapMessages_EmptyList() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        assertEquals("[]", result.get("messages"));
    }

    @Test
    public void testMapMessages_NullList() {
        // Act
        Map<String, String> result = provider.mapMessages(null, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        assertEquals("[]", result.get("messages"));
    }

    @Test
    public void testMapAgentInput_TextInput() {
        // Arrange
        String text = "Hello, how are you?";
        AgentInput agentInput = new AgentInput();
        agentInput.setInput(text);

        // Act
        Map<String, String> result = provider.mapAgentInput(agentInput, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"role\":\"user\""));
        assertTrue(messages.contains("Hello, how are you?"));
    }

    @Test
    public void testMapAgentInput_ContentBlocksInput() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Test content");
        blocks.add(textBlock);

        AgentInput agentInput = new AgentInput();
        agentInput.setInput(blocks);

        // Act
        Map<String, String> result = provider.mapAgentInput(agentInput, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messages = result.get("messages");
        assertTrue(messages.contains("\"text\":\"Test content\""));
    }

    @Test
    public void testMapAgentInput_MessagesInput() {
        // Arrange
        List<Message> messages = new ArrayList<>();

        List<ContentBlock> content = new ArrayList<>();
        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Hello");
        content.add(textBlock);

        Message message = new Message("user", content);
        messages.add(message);

        AgentInput agentInput = new AgentInput();
        agentInput.setInput(messages);

        // Act
        Map<String, String> result = provider.mapAgentInput(agentInput, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"role\":\"user\""));
        assertTrue(messagesStr.contains("Hello"));
    }

    @Test
    public void testMapAgentInput_NullAgentInput() {
        // Act & Assert
        try {
            provider.mapAgentInput(null, MLAgentType.CONVERSATIONAL);
            fail("Should throw IllegalArgumentException for null AgentInput");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("AgentInput and its input field cannot be null"));
        }
    }

    @Test
    public void testMapAgentInput_NullInputField() {
        // Arrange
        AgentInput agentInput = new AgentInput();
        agentInput.setInput(null);

        // Act & Assert
        try {
            provider.mapAgentInput(agentInput, MLAgentType.CONVERSATIONAL);
            fail("Should throw IllegalArgumentException for null input field");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("AgentInput and its input field cannot be null"));
        }
    }

    @Test
    public void testFormatImageDataUrl_WithDefaultFormat() {
        // This is an indirect test through mapContentBlocks
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, null, "testdata");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        String messages = result.get("messages");
        assertTrue(messages.contains("data:image/jpeg;base64,testdata")); // default format is jpeg
    }

    @Test
    public void testFormatImageDataUrl_WithCustomFormat() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();
        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, "webp", "webpdata");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        String messages = result.get("messages");
        assertTrue(messages.contains("data:image/webp;base64,webpdata"));
    }

    @Test
    public void testCreateConnector_VerifyHeaders() {
        // Arrange
        String modelId = "gpt-4";
        Map<String, String> credential = new HashMap<>();
        credential.put("openAI_key", "sk-test");

        // Act
        Connector connector = provider.createConnector(modelId, credential, null);

        // Assert
        HttpConnector httpConnector = (HttpConnector) connector;
        assertNotNull(httpConnector.getActions());
        assertEquals(1, httpConnector.getActions().size());
        
        Map<String, String> headers = httpConnector.getActions().get(0).getHeaders();
        assertNotNull(headers);
        assertTrue(headers.containsKey("Content-Type"));
        assertEquals("application/json", headers.get("Content-Type"));
        assertTrue(headers.containsKey("Authorization"));
        assertTrue(headers.get("Authorization").contains("Bearer"));
    }

    @Test
    public void testCreateConnector_VerifyURL() {
        // Arrange
        String modelId = "gpt-4";
        Map<String, String> credential = new HashMap<>();
        credential.put("openAI_key", "sk-test");

        // Act
        Connector connector = provider.createConnector(modelId, credential, null);

        // Assert
        HttpConnector httpConnector = (HttpConnector) connector;
        String url = httpConnector.getActions().get(0).getUrl();
        assertTrue(url.contains("https://"));
        assertTrue(url.contains("/v1/chat/completions"));
        assertTrue(url.contains("${parameters.endpoint}"));
    }

    @Test
    public void testMapMessages_EmptyContentInMessage() {
        // Arrange
        List<Message> messages = new ArrayList<>();
        Message message = new Message("user", new ArrayList<>());
        messages.add(message);

        // Act
        Map<String, String> result = provider.mapMessages(messages, MLAgentType.CONVERSATIONAL);

        // Assert
        assertNotNull(result);
        assertTrue(result.containsKey("messages"));
        String messagesStr = result.get("messages");
        assertTrue(messagesStr.contains("\"role\":\"user\""));
        assertTrue(messagesStr.contains("\"\""));
    }

    @Test
    public void testMapContentBlocks_MixedSupportedAndUnsupportedTypes() {
        // Arrange
        List<ContentBlock> blocks = new ArrayList<>();

        ContentBlock textBlock = new ContentBlock();
        textBlock.setType(ContentType.TEXT);
        textBlock.setText("Some text");
        blocks.add(textBlock);

        ContentBlock docBlock = new ContentBlock();
        docBlock.setType(ContentType.DOCUMENT);
        DocumentContent doc = new DocumentContent(SourceType.BASE64, "pdf", "pdfdata");
        docBlock.setDocument(doc);
        blocks.add(docBlock);

        ContentBlock imageBlock = new ContentBlock();
        imageBlock.setType(ContentType.IMAGE);
        ImageContent image = new ImageContent(SourceType.BASE64, "png", "imagedata");
        imageBlock.setImage(image);
        blocks.add(imageBlock);

        // Act
        Map<String, String> result = provider.mapContentBlocks(blocks, MLAgentType.CONVERSATIONAL);

        // Assert
        String messages = result.get("messages");
        assertTrue(messages.contains("Some text"));
        assertTrue(messages.contains("data:image/png;base64,imagedata"));
        // Document should be skipped
    }
}
