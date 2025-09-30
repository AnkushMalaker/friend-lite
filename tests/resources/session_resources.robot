*** Settings ***
Documentation    API session creation and authentication management keywords
...
...              This file contains keywords for API session management and authentication.
...              Keywords in this file should handle session creation, authentication workflows,
...              token management, and session cleanup.
...
...              Examples of keywords that belong here:
...              - API session creation and management
...              - Authentication workflows
...              - Token extraction (when needed for external tools)
...              - Session validation and cleanup
...
...              Keywords that should NOT be in this file:
...              - Verification/assertion keywords (belong in tests)
...              - User management operations (belong in user_resources.robot)
...              - Docker service management (belong in setup_resources.robot)
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py

*** Keywords ***

# Core Session Creation
Create API Session
    [Documentation]    Create an API session (authenticated or anonymous)
    [Arguments]        ${session_name}    ${email}=${ADMIN_EMAIL}    ${password}=${ADMIN_PASSWORD}  ${base_url}=${API_URL}

    # Create base session
    Create Session    ${session_name}    ${base_url}    verify=True


    ${token}=    Get Authentication Token    ${session_name}    ${email}    ${password}
     &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    # Update session with auth headers
    Create Session    ${session_name}    ${base_url}    verify=True    headers=${headers}
    Set Suite Variable    ${session_name}

Get Anonymous Session
    [Documentation]    Get an unauthenticated API session
    [Arguments]    ${session_name}    ${base_url}=${API_URL}

    Create Session    ${session_name}    ${base_url}    verify=True


# Core Authentication
Get Authentication Token
    [Documentation]    Get authentication token for any user from existing session
    [Arguments]    ${session_alias}    ${email}    ${password}

    &{auth_data}=    Create Dictionary    username=${email}    password=${password}
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    ${session_alias}    /auth/jwt/login    data=${auth_data}    headers=${headers}    expected_status=200

    ${json_response}=    Set Variable    ${response.json()}
    ${token}=    Get From Dictionary    ${json_response}    access_token
    RETURN    ${token}


Get Current User From Session
    [Documentation]    Get current user information from authenticated session
    [Arguments]    ${session_alias}

    ${response}=    GET On Session    ${session_alias}    /users/me    expected_status=any
    RETURN    ${response}
