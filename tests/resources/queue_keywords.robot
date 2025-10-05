*** Settings ***
Documentation    Memory Management Keywords
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py
Resource         session_resources.robot

*** Keywords ***

Get job queue
    [Documentation]    Get the current job queue from Redis
    [Arguments]    ${queue_name}=default
    ${response}=    GET On Session    api    /api/queue/jobs
    ${jobs}=    Set Variable    ${response.json()}[jobs]
    RETURN    ${jobs}

Get queue length
    [Documentation]    Get the length of the specified job queue
    [Arguments]    ${queue_name}=default
    ${jobs}=     Get job queue     ${queue_name}
    ${length}=    Get Length    ${jobs}
    RETURN    ${length}

Get Job Details
    [Documentation]    Get job details from the queue API by searching the jobs list
    [Arguments]    ${job_id}

    ${response}=    GET On Session    api    /api/queue/jobs
    Should Be Equal As Integers    ${response.status_code}    200
    ${jobs_data}=    Set Variable    ${response.json()}
    ${jobs}=    Set Variable    ${jobs_data}[jobs]

    # Find the job with matching job_id
    FOR    ${job}    IN    @{jobs}
        IF    '${job}[job_id]' == '${job_id}'
            RETURN    ${job}
        END
    END

Check job status
    [Documentation]    Check the status of a specific job by ID
    [Arguments]    ${job_id}    ${expected_status}
    ${job}=    Get Job Details    ${job_id}
    ${actual_status}=    Set Variable    ${job}[status]
    Should Be Equal As Strings    ${actual_status}    ${expected_status}    Job status does not match expected status
    RETURN    ${job}

    # If we get here, job not found
    Fail    Job with ID ${job_id} not found in queue
Clear job queue
    [Documentation]    Clear all jobs from the specified queue
    [Arguments]    ${queue_name}=default
    ${response}=    DELETE On Session    api    /api/queue/jobs
    RETURN    ${response}

Clear job by ID
    [Documentation]    Clear a specific job by ID
    [Arguments]    ${job_id}

    ${response}=    DELETE On Session    api    /api/queue/jobs/${job_id}
    RETURN    ${response}

Enqueue test job
    [Documentation]    Enqueue a test job into the specified queue
    [Arguments]    ${queue_name}=default    ${job_data}={}

    &{data}=    Create Dictionary    queue=${queue_name}    job_data=${job_data}
    ${response}=    POST On Session    api    /api/queue/enqueue    json=${data}
    RETURN    ${response}