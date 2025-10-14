*** Settings ***
Documentation    Conversation Queue Integration Tests
Library          RequestsLibrary
Library          Collections
Resource         ../resources/setup_resources.robot
Resource         ../resources/session_resources.robot
Resource         ../resources/audio_keywords.robot
Resource         ../resources/conversation_keywords.robot
Resource         ../resources/queue_keywords.robot
Variables        ../test_env.py
Variables        ../test_data.py
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions
Test Setup       Clear Test Databases


*** Test Cases ***

Test Upload audio creates transcription job
    [Documentation]    Test that uploading audio creates a transcription job in the queue
    [Tags]             integration    queue    upload
    [Timeout]          120s

    Log    Starting Upload Job Queue Test    INFO

    # Verify queue is empty
    ${initial_job_count}=    Get queue length
    Should Be Equal As Integers    ${initial_job_count}    0

    # Upload audio file to create conversation and trigger transcription job
    ${conversation}=    Upload Audio File   ${TEST_AUDIO_FILE}    ${TEST_DEVICE_NAME}
    ${conversation_id}=    Set Variable    ${conversation}[conversation_id]

    Log    Created conversation: ${conversation_id}    INFO

    # Verify a new job has been added to the queue
    Wait Until Keyword Succeeds    10s    2s    Get queue length
    ${job_count}=    Get queue length
    Should Be True    ${job_count} >= 1    Expected at least 1 job in queue, got ${job_count}

    # Get the list of jobs and find the transcription job for our conversation
    ${jobs}=    Get job queue

    # Find the transcription job for our conversation
    # Note: transcript jobs have job_type "reprocess_transcript" and conversation_id as args[0]
    ${transcription_job}=    Set Variable    None
    FOR    ${job}    IN    @{jobs}
        ${job_type}=    Set Variable    ${job}[job_type]
        # Check if this is a transcript job (job_type contains "transcript")
        ${is_transcript_job}=    Evaluate    "transcript" in "${job_type}".lower()
        IF    ${is_transcript_job}
            # Get conversation_id from args[0] (first argument to transcript job)
            ${job_conv_id}=    Set Variable    ${job}[args][0]
            # Check if conversation_id matches (compare first 8 chars for short IDs)
            ${conv_id_short}=    Evaluate    "${conversation_id}"[:8]
            ${job_conv_id_short}=    Evaluate    "${job_conv_id}"[:8]
            IF    '${conv_id_short}' == '${job_conv_id_short}'
                ${transcription_job}=    Set Variable    ${job}
                Exit For Loop
            END
        END
    END
    Should Not Be Equal    ${transcription_job}    None    Transcription job for conversation ${conversation_id} not found in queue

Test Reprocess Conversation Job Queue
    [Documentation]    Test that reprocess transcript jobs are created and processed correctly
    [Tags]             integration    queue    reprocess
    [Timeout]          180s

    Log    Starting Reprocess Job Queue Test    INFO

    # First, create a conversation by uploading audio
    ${conversation}=    Upload Audio File   ${TEST_AUDIO_FILE}    ${TEST_DEVICE_NAME}
    ${conversation_id}=    Set Variable    ${conversation}[conversation_id]

    Log    Created conversation: ${conversation_id}    INFO

    # verify existing jobs to get clean baseline
    ${initial_job_count}=    Get queue length
    Log To Console    Initial job count: ${initial_job_count}    INFO

    # Trigger transcript reprocessing
    Log    Triggering transcript reprocessing for conversation ${conversation_id}    INFO
    ${reprocess_data}=    Reprocess Transcript    ${conversation_id}
    ${job_id}=    Set Variable    ${reprocess_data}[job_id]
    # Verify job is not failed initially
    Should Not Be Equal As Strings    "failed"    ${reprocess_data}[status]

    Sleep    2s    # Give RQ workers a moment to pick up the job
    ${job}=    Get Job Details    ${job_id}

    # Wait for job to be processed with timeout (RQ workers need time for real transcription)
    Wait Until Keyword Succeeds    60s    2s    Job Should Be Complete       ${job_id}
    ${job_details}=    Get Job Details  ${job_id}

    # Verify job structure matches UI expectations
    Dictionary Should Contain Key    ${job_details}    job_id
    Dictionary Should Contain Key    ${job_details}    job_type
    Dictionary Should Contain Key    ${job_details}    status
    Dictionary Should Contain Key    ${job_details}    priority
    Dictionary Should Contain Key    ${job_details}    created_at
    Dictionary Should Contain Key    ${job_details}    queue_name

    # Verify job details
    Should Be Equal As Strings    ${job_details}[job_id]    ${job_id}
    Should Be Equal As Strings    ${job_details}[job_type]    reprocess_transcript
    Should Be Equal As Strings    ${job_details}[queue_name]    transcription
    Should Be Equal As Strings    ${job_details}[priority]    normal

    # Job should be completed (or failed) by now
    Should Be True    '${job_details}[status]' in ['completed', 'finished']    Job status: ${job_details}[status], expected completed or finished

    # Log job details for debugging
    Log    Job details: ${job_details}    INFO

    # Check if job has result data (some completed jobs might not have result field)
    ${has_result}=    Run Keyword And Return Status    Dictionary Should Contain Key    ${job_details}    result
    IF    ${has_result}
        ${result}=    Set Variable    ${job_details}[result]
        Log    Job result: ${result}    INFO
        Dictionary Should Contain Key    ${result}    success
        Dictionary Should Contain Key    ${result}    conversation_id
        Should Be Equal As Strings    ${result}[conversation_id]    ${conversation_id}
    ELSE
        Log    Job completed but has no result field - checking conversation was updated    WARN
    END

    # Verify conversation was actually updated with new transcript version
    ${updated_conversation}=    Get Conversation By ID    ${conversation_id}

    # Check that version info shows multiple transcript versions
    Dictionary Should Contain Key    ${updated_conversation}    version_info
    ${version_info}=    Set Variable    ${updated_conversation}[version_info]
    Dictionary Should Contain Key    ${version_info}    transcript_count
    ${transcript_count}=    Set Variable    ${version_info}[transcript_count]
    Should Be True    ${transcript_count} > 1    Expected multiple transcript versions, got ${transcript_count}
    Should Be True   ${updated_conversation}[transcript] != []    Expected conversation to have transcript after reprocessing

    Log    Reprocess Job Queue Test Completed Successfully    INFO

*** Keywords ***

Job Should Be Complete
    [Documentation]    Check if job has reached a completed state
    [Arguments]     ${job_id}

    ${job_details}=    Get Job Details    ${job_id}
    ${status}=    Set Variable    ${job_details}[status]
    Should Be True    '${status}' in ['completed', 'finished', 'failed']    Job status: ${status}
