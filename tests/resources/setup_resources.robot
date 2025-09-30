*** Settings ***
Documentation    Reusable keywords for API testing
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Library          String
Library          Process
Variables        ../test_env.py
Resource         ../resources/session_resources.robot


*** Keywords ***

Suite Setup
    [Documentation]    Setup for auth test suite
    ${random_id}=    Generate Random String    8    [LETTERS][NUMBERS]
    Set Suite Variable    ${RANDOM_ID}    ${random_id}
    Start advanced-server
    Create API session    api

Suite Teardown
    # Stop and remove containers with volumes
    Run   docker-compose -f backends/advanced/docker-compose-test.yml down -v
    # Clean up any remaining volumes
    Run    rm -rf backends/advanced/data/test_mongo_data
    Run    rm -rf ${EXECDIR}/backends/advanced/data/test_qdrant_data
    Run    rm -rf ${EXECDIR}/backends/advanced/data/test_audio_chunks
    Delete All Sessions

Start advanced-server
    [Documentation]    Start the server using docker-compose
    ${is_up}=    Run Keyword And Return Status    Readiness Check    ${API_URL}
    IF    ${is_up}
        Log    advanced-server is already running
        RETURN
    ELSE
        Log    Starting advanced-server
        Run    docker-compose -f backends/advanced/docker-compose-test.yml up -d --build
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

Clear Test Databases
    [Documentation]    Quickly clear test databases and audio files without restarting containers (preserves admin user)
    Log To Console    Clearing test databases and audio files...

    # Clear MongoDB collections but preserve admin user
    Run    docker exec advanced-mongo-test-1 mongo test_db --eval "db.users.deleteMany({'email': {$ne:'${ADMIN_EMAIL}'}})"
    Run    docker exec advanced-mongo-test-1 mongo test_db --eval "db.conversations.deleteMany({})"
    Run    docker exec advanced-mongo-test-1 mongo test_db --eval "db.audio_chunks.deleteMany({})"
    # Clear admin user's registered_clients array to prevent client_id counter increments
    Run    docker exec advanced-mongo-test-1 mongo test_db --eval "db.users.updateOne({'email':'${ADMIN_EMAIL}'}, {$set: {'registered_clients': []}})"
    Log To Console    MongoDB collections cleared (except admin user)

    # Clear Qdrant collections
    Run    curl -s -X DELETE http://localhost:6337/collections/memories
    Run    curl -s -X DELETE http://localhost:6337/collections/conversations
    Log To Console    Qdrant collections cleared

    # Clear audio files from mounted volumes
    Run    rm -rf ${EXECDIR}/backends/advanced/data/test_audio_chunks/*
    Run    rm -rf ${EXECDIR}/backends/advanced/data/test_debug_dir/*
    # Also clear any files inside the container (in case of different mount paths)
    Run    docker exec advanced-friend-backend-test-1 find /app/audio_chunks -name "*.wav" -delete 2>/dev/null || true
    Run    docker exec advanced-friend-backend-test-1 find /app/debug_dir -name "*" -type f -delete 2>/dev/null || true
    Log To Console    Audio files and debug files cleared

    # Clear Redis queues and job registries
    Run    docker exec advanced-redis-test-1 redis-cli FLUSHALL
    Log To Console    Redis queues and job registries cleared