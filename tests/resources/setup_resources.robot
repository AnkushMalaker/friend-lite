*** Settings ***
Documentation    Reusable keywords for API testing
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Library          String
Variables        ../test_env.py
Resource        login_resources.robot


*** Keywords ***

Suite Setup
    [Documentation]    Setup for auth test suite
    ${random_id}=    Generate Random String    8    [LETTERS][NUMBERS]
    Set Suite Variable    ${RANDOM_ID}    ${random_id}

    # Ensure server is running
    Health Check    ${API_URL}

Start advanced-server
    [Documentation]    Start the server using docker-compose
    Run    docker-compose -f docker-compose.test.yml up -d --build
    Sleep    5s
    Health Check    ${API_URL}

Stop advanced-server
    [Documentation]    Stop the server using docker-compose
    Run    docker-compose -f docker-compose.test.yml down

Health Check
    [Documentation]    Verify that the health endpoint is accessible
    [Tags]             health    api
    [Arguments]        ${base_url}= ${API_URL}
    Wait Until Keyword Succeeds    30s    5s    GET    ${base_url}/health
        ${response}=    GET    ${base_url}/health
        Should Be Equal As Integers    ${response.status_code}    200

Setup API Session
    [Documentation]    Create session for API testing
    [Arguments]    ${base_url}=http://localhost:8001    &{headers}
    IF    ${headers} == '{}'
        &{headers}=    Create Basic Auth Header     ${ADMIN_EMAIL}    ${ADMIN_PASSWORD}
    END
    Create Session    api    ${base_url}    verify=True headers=${headers}
    RETURN    api

# Get Auth Token
#     [Documentation]    Get authentication token for API calls
#     [Arguments]    ${email}    ${password}

#     &{auth_data}=    Create Dictionary    username=${email}    password=${password}
#     &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

#     ${response}=    POST On Session    api    /auth/jwt/login    data=${auth_data}    headers=${headers}
#     Should Be Equal As Integers    ${response.status_code}    200

#     ${token}=    Get Value From Json    ${response.json()}    $.access_token
#     RETURN    ${token}[0]

# Make Authenticated Request
#     [Documentation]    Make an authenticated API request
#     [Arguments]    ${method}    ${endpoint}    ${token}    &{kwargs}

#     &{headers}=    Create Dictionary    Authorization=Bearer ${token}
#     Set To Dictionary    ${kwargs}    headers=${headers}

#     ${response}=    Run Keyword    ${method} On Session    api    ${endpoint}    &{kwargs}
#     RETURN    ${response}

# Verify JSON Response Contains
#     [Documentation]    Verify JSON response contains expected data
#     [Arguments]    ${response}    ${key}    ${expected_value}=None

#     ${json_data}=    Set Variable    ${response.json()}
#     Should Contain    ${json_data}    ${key}

#     IF    '${expected_value}' != 'None'
#         Should Be Equal    ${json_data}[${key}]    ${expected_value}
#     END