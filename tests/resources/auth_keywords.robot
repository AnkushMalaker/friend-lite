*** Settings ***
Documentation    Authentication and User Management Keywords
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py

*** Keywords ***

Setup Auth Session
    [Documentation]    Create API session for authentication tests
    [Arguments]    ${base_url}=${API_URL}
    Create Session    api    ${base_url}    verify=True
    RETURN    api

Get Admin Token
    [Documentation]    Get authentication token for admin user
    &{auth_data}=    Create Dictionary    username=${ADMIN_EMAIL}    password=${ADMIN_PASSWORD}
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    api    /auth/jwt/login    data=${auth_data}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${json_response}=    Set Variable    ${response.json()}
    ${token}=    Get From Dictionary    ${json_response}    access_token
    RETURN    ${token}

Verify Admin User
    [Documentation]    Verify current user is admin
    [Arguments]    ${token}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    ${response}=    GET On Session    api    /users/me    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200
    Should Be Equal    ${response.json()}[email]    ${ADMIN_EMAIL}
    RETURN    ${response.json()}

Create Test User
    [Documentation]    Create a test user for testing
    [Arguments]    ${token}    ${email}    ${password}    ${is_superuser}=False
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json
    &{user_data}=    Create Dictionary    email=${email}    password=${password}    is_superuser=${is_superuser}

    ${response}=    POST On Session    api    /api/users    json=${user_data}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    201
    RETURN    ${response.json()}

Delete Test User
    [Documentation]    Delete a test user
    [Arguments]    ${token}    ${user_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    ${response}=    DELETE On Session    api    /api/users/${user_id}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

Get User Token
    [Documentation]    Get authentication token for any user
    [Arguments]    ${email}    ${password}
    &{auth_data}=    Create Dictionary    username=${email}    password=${password}
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    api    /auth/jwt/login    data=${auth_data}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${json_response}=    Set Variable    ${response.json()}
    ${token}=    Get From Dictionary    ${json_response}    access_token
    RETURN    ${token}

Make Authenticated Request
    [Documentation]    Make an authenticated API request
    [Arguments]    ${method}    ${endpoint}    ${token}    &{kwargs}

    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    Set To Dictionary    ${kwargs}    headers=${headers}

    ${response}=    Run Keyword    ${method} On Session    api    ${endpoint}    &{kwargs}
    RETURN    ${response}

Verify Unauthorized Access
    [Documentation]    Verify endpoint requires authentication
    [Arguments]    ${method}    ${endpoint}    &{kwargs}
    ${response}=    Run Keyword And Expect Error    *    Run Keyword    ${method} On Session    api    ${endpoint}    expected_status=401    &{kwargs}

Verify Admin Required
    [Documentation]    Verify endpoint requires admin privileges
    [Arguments]    ${method}    ${endpoint}    ${non_admin_token}    &{kwargs}
    &{headers}=    Create Dictionary    Authorization=Bearer ${non_admin_token}
    Set To Dictionary    ${kwargs}    headers=${headers}
    ${response}=    Run Keyword    ${method} On Session    api    ${endpoint}    expected_status=403    &{kwargs}
    Should Be Equal As Integers    ${response.status_code}    403