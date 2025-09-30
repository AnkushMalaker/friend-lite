*** Settings ***
Documentation    Advanced Transcript Verification Keywords
...              Includes OpenAI-powered similarity checking similar to the Python integration tests
Library          RequestsLibrary
Library          Collections
Library          String
# Library          JSONLibrary  # Optional library, not required

*** Variables ***
${OPENAI_API_BASE}           https://api.openai.com/v1
${OPENAI_MODEL}              gpt-4o-mini
${SIMILARITY_THRESHOLD}      0.7
${EXPECTED_GROUND_TRUTH}     experts in glass blowing demonstrating techniques

*** Keywords ***
Verify Transcript With AI Similarity
    [Documentation]    Use OpenAI to verify transcript similarity to ground truth
    [Arguments]    ${transcript}    ${ground_truth}=${EXPECTED_GROUND_TRUTH}    ${threshold}=${SIMILARITY_THRESHOLD}

    # Get OpenAI API key
    ${openai_key}=    Get Environment Variable    OPENAI_API_KEY    ${EMPTY}
    Should Not Be Empty    ${openai_key}    OPENAI_API_KEY required for AI similarity checking

    # Prepare similarity check prompt
    ${prompt}=    Create Similarity Check Prompt    ${transcript}    ${ground_truth}

    # Call OpenAI API
    ${similarity_score}=    Get Transcript Similarity Score    ${openai_key}    ${prompt}

    # Validate similarity
    Should Be True    ${similarity_score} >= ${threshold}    Transcript similarity ${similarity_score} below threshold ${threshold}

    Log    Transcript similarity verification passed: ${similarity_score}    INFO
    RETURN    ${similarity_score}

Create Similarity Check Prompt
    [Documentation]    Create prompt for OpenAI similarity checking
    [Arguments]    ${transcript}    ${ground_truth}

    ${prompt}=    Catenate    SEPARATOR=\n
    ...    You are evaluating the similarity between a speech-to-text transcript and ground truth content.
    ...
    ...    Ground Truth: "${ground_truth}"
    ...    Transcript: "${transcript}"
    ...
    ...    Rate the semantic similarity on a scale of 0.0 to 1.0, where:
    ...    - 1.0 = Perfect semantic match
    ...    - 0.8+ = Very similar meaning, minor differences
    ...    - 0.6+ = Generally similar topics and concepts
    ...    - 0.4+ = Some related content
    ...    - 0.0 = Completely unrelated
    ...
    ...    Focus on meaning and content, not exact word matching.
    ...    Consider that speech-to-text may have minor transcription errors.
    ...
    ...    Respond with just the numerical score (e.g., "0.85").

    RETURN    ${prompt}

Get Transcript Similarity Score
    [Documentation]    Call OpenAI API to get similarity score
    [Arguments]    ${api_key}    ${prompt}

    # Prepare request
    Create Session    openai    ${OPENAI_API_BASE}
    &{headers}=    Create Dictionary
    ...    Authorization=Bearer ${api_key}
    ...    Content-Type=application/json

    &{request_data}=    Create Dictionary
    ...    model=${OPENAI_MODEL}
    ...    messages=${[{"role": "user", "content": "${prompt}"}]}
    ...    max_tokens=10
    ...    temperature=0.1

    # Make API call
    ${response}=    POST On Session    openai    /chat/completions    headers=${headers}    json=${request_data}    expected_status=200

    # Parse response
    ${response_data}=    Set Variable    ${response.json()}
    ${content}=          Set Variable    ${response_data}[choices][0][message][content]
    ${score_text}=       Strip String    ${content}

    # Convert to float
    TRY
        ${similarity_score}=    Convert To Number    ${score_text}
        Delete All Sessions    openai
        RETURN    ${similarity_score}
    EXCEPT
        Delete All Sessions    openai
        Fail    Invalid similarity score from OpenAI: ${score_text}
    END

Verify Transcript Quality Metrics
    [Documentation]    Verify various transcript quality metrics
    [Arguments]    ${conversation}    ${expected_keywords}    ${min_length}=100

    Dictionary Should Contain Key    ${conversation}    transcript
    ${transcript}=    Set Variable    ${conversation}[transcript]

    # Basic quality checks
    Should Not Be Empty    ${transcript}
    ${length}=    Get Length    ${transcript}
    Should Be True    ${length} >= ${min_length}    Transcript too short: ${length} chars

    # Check for expected keywords
    ${transcript_lower}=    Convert To Lower Case    ${transcript}
    FOR    ${keyword}    IN    @{expected_keywords}
        ${keyword_lower}=    Convert To Lower Case    ${keyword}
        Should Contain    ${transcript_lower}    ${keyword_lower}    Missing keyword: ${keyword}
    END

    # Segment validation
    Dictionary Should Contain Key    ${conversation}    segments
    ${segments}=    Set Variable    ${conversation}[segments]
    ${segment_count}=    Get Length    ${segments}
    Should Be True    ${segment_count} > 0    No segments found

    # Validate segment structure
    FOR    ${segment}    IN    @{segments}
        Dictionary Should Contain Key    ${segment}    start
        Dictionary Should Contain Key    ${segment}    end
        Dictionary Should Contain Key    ${segment}    text
        Should Be True    ${segment}[end] > ${segment}[start]    Invalid segment timing
    END

    # Quality heuristics
    ${word_count}=    Get Word Count    ${transcript}
    Should Be True    ${word_count} >= 20    Too few words: ${word_count}

    # Check for common transcription errors/patterns
    ${error_patterns}=    Create List    [inaudible]    [unclear]    ***    ERROR    FAILED
    FOR    ${pattern}    IN    @{error_patterns}
        Should Not Contain    ${transcript_lower}    ${pattern}    Transcript contains error pattern: ${pattern}
    END

    Log    Transcript quality metrics passed: ${length} chars, ${word_count} words, ${segment_count} segments    INFO

Get Word Count
    [Documentation]    Count words in text
    [Arguments]    ${text}

    ${words}=    Split String    ${text}
    ${count}=    Get Length    ${words}
    RETURN    ${count}

Calculate Transcript Statistics
    [Documentation]    Calculate detailed transcript statistics
    [Arguments]    ${conversation}

    ${transcript}=    Set Variable    ${conversation}[transcript]
    ${segments}=      Set Variable    ${conversation}[segments]

    # Basic statistics
    ${char_count}=     Get Length    ${transcript}
    ${word_count}=     Get Word Count    ${transcript}
    ${segment_count}=  Get Length    ${segments}

    # Timing statistics
    ${total_duration}=    Calculate Total Duration    ${segments}
    ${speech_rate}=       Evaluate    ${word_count} / (${total_duration} / 60) if ${total_duration} > 0 else 0

    # Create statistics dictionary
    &{stats}=    Create Dictionary
    ...    character_count=${char_count}
    ...    word_count=${word_count}
    ...    segment_count=${segment_count}
    ...    total_duration_seconds=${total_duration}
    ...    words_per_minute=${speech_rate}

    Log    Transcript statistics: ${stats}    INFO
    RETURN    &{stats}

Calculate Total Duration
    [Documentation]    Calculate total duration from segments
    [Arguments]    ${segments}

    ${total}=    Set Variable    0
    FOR    ${segment}    IN    @{segments}
        ${duration}=    Evaluate    ${segment}[end] - ${segment}[start]
        ${total}=       Evaluate    ${total} + ${duration}
    END
    RETURN    ${total}

Verify Segment Speaker Diarization
    [Documentation]    Verify speaker diarization in segments
    [Arguments]    ${segments}    ${expect_multiple_speakers}=${False}

    ${speaker_ids}=    Create List
    FOR    ${segment}    IN    @{segments}
        IF    'speaker' in ${segment}
            ${speaker_id}=    Set Variable    ${segment}[speaker]
            ${contains}=      Evaluate    $speaker_id in $speaker_ids
            IF    not ${contains}
                Append To List    ${speaker_ids}    ${speaker_id}
            END
        END
    END

    ${speaker_count}=    Get Length    ${speaker_ids}

    IF    ${expect_multiple_speakers}
        Should Be True    ${speaker_count} > 1    Expected multiple speakers, found ${speaker_count}
    ELSE
        Should Be True    ${speaker_count} >= 1    No speakers identified
    END

    Log    Speaker diarization: ${speaker_count} unique speakers found    INFO
    RETURN    ${speaker_count}