*** Settings ***
Documentation    Reusable keywords for API testing
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Library          String
Variables        ../test_env.py
Resource         ../resources/session_resources.robot


*** Keywords ***

Suite Setup
    [Documentation]    Setup for auth test suite
    ${random_id}=    Generate Random String    8    [LETTERS][NUMBERS]
    Set Suite Variable    ${RANDOM_ID}    ${random_id}
    Start advanced-server
    Create API session    ${API_URL}

Start advanced-server
    [Documentation]    Start the server using docker-compose
    ${is_up}=    Run Keyword And Return Status    Readiness Check    ${API_URL}
    IF    ${is_up}
        Log    advanced-server is already running
        RETURN
    ELSE
        Log    Starting advanced-server
        Run    docker-compose -f ../../backends/advanced/docker-compose-test.yml up -d --build
        Log    Waiting for services to start...
        Wait Until Keyword Succeeds    60s    5s    Readiness Check    ${API_URL}
        Log    Services are ready
    END
    
Stop advanced-server
    [Documentation]    Stop the server using docker-compose
    Run    docker-compose -f docker-compose.test.yml down

Start speaker-recognition-service
    [Documentation]    Start the speaker recognition service using docker-compose
    ${is_up}=    Run Keyword And Return Status    Readiness Check    ${SPEAKER_RECOGNITION_URL}
    IF    ${is_up}
        Log    speaker-recognition-service is already running    
        RETURN
    ELSE
        Log    Starting speaker-recognition-service
        Run    docker-compose -f ../../extras/speaker_recognition/docker-compose.test.yml up -d --build
        Log    Waiting for speaker recognition service to start...
        Wait Until Keyword Succeeds    60s    5s    Readiness Check    ${SPEAKER_RECOGNITION_URL}
        Log    Speaker recognition service is ready
    END

Readiness Check
    [Documentation]    Verify that the readiness endpoint is accessible (faster than /health)
    [Tags]             health    api
    [Arguments]        ${base_url}=${API_URL}

    ${response}=    GET    ${base_url}/readiness    expected_status=200
    Should Be Equal As Integers    ${response.status_code}    200
    RETURN    ${True}

Health Check
    [Documentation]    Verify that the readiness endpoint is accessible (faster than /health)
    [Tags]             health    api
    [Arguments]        ${base_url}=${API_URL}

    ${response}=    GET    ${base_url}/health    expected_status=200
    Should Be Equal As Integers    ${response.status_code}    200
    RETURN    ${True}

