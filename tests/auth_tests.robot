*** Settings ***
Documentation    Authentication and User Management API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Library          BuiltIn
Resource         resources/setup_resources.robot
Resource         resources/auth_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Variables ***
# Test users are now imported from test_env.py via resource files

*** Test Cases ***

Login With Valid Credentials Test
    [Documentation]    Test successful login with admin credentials
    [Tags]             auth    login    positive
    Setup Auth Session

    ${token}=    Get Admin Token
    Should Not Be Empty    ${token}
    Log    Successfully obtained admin token

    # Verify token works
    ${user}=    Verify Admin User    ${token}
    Should Be Equal    ${user}[email]    ${ADMIN_EMAIL}

Login With Invalid Credentials Test
    [Documentation]    Test login failure with invalid credentials
    [Tags]             auth    login    negative
    Setup Auth Session

    &{auth_data}=    Create Dictionary    username=${ADMIN_EMAIL}    password=wrong-password
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    api    /auth/jwt/login    data=${auth_data}    headers=${headers}    expected_status=400
    Should Be Equal As Integers    ${response.status_code}    400

Get Current User Test
    [Documentation]    Test getting current authenticated user
    [Tags]             auth    user    positive
    Setup Auth Session

    ${token}=    Get Admin Token
    ${user}=     Verify Admin User    ${token}

    Dictionary Should Contain Key    ${user}    email
    Dictionary Should Contain Key    ${user}    id
    Should Be Equal       ${user}[email]    ${ADMIN_EMAIL}

Unauthorized Access Test
    [Documentation]    Test that endpoints require authentication
    [Tags]             auth    security    negative
    Setup Auth Session

    # Try to access protected endpoint without token
    ${response}=    GET On Session    api    /users/me    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Create User Test
    [Documentation]    Test creating a new user (admin only)
    [Tags]             users    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User    ${admin_token}    ${test_email}    ${TEST_USER_PASSWORD}

    Should Be Equal    ${user}[user_email]    ${test_email}
    Should Contain     ${user}[message]    created successfully

    # Cleanup
    Delete Test User    ${admin_token}    ${user}[user_id]

Create Admin User Test
    [Documentation]    Test creating an admin user
    [Tags]             users    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    ${test_admin_email}=    Set Variable    test-admin-${RANDOM_ID}@example.com
    ${user}=           Create Test User    ${admin_token}    ${test_admin_email}    ${TEST_USER_PASSWORD}    ${True}

    Should Be Equal    ${user}[user_email]    ${test_admin_email}
    Should Contain     ${user}[message]    created successfully

    # Cleanup
    Delete Test User    ${admin_token}    ${user}[user_id]

Get All Users Test
    [Documentation]    Test getting all users (admin only)
    [Tags]             users    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    ${response}=       Make Authenticated Request    GET    /api/users    ${admin_token}

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True     isinstance($response.json(), list)

    # Should contain at least the admin user
    ${users}=    Set Variable    ${response.json()}
    ${admin_found}=    Set Variable    ${False}
    FOR    ${user}    IN    @{users}
        IF    '${user}[email]' == '${ADMIN_EMAIL}'
            ${admin_found}=    Set Variable    ${True}
        END
    END
    Should Be True    ${admin_found}

Non-Admin User Cannot Create Users Test
    [Documentation]    Test that non-admin users cannot create users
    [Tags]             users    security    negative
    Setup Auth Session

    ${admin_token}=    Get Admin Token

    # Create a non-admin user
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${test_user}=      Create Test User    ${admin_token}    ${test_email}    ${TEST_USER}[password]
    ${user_token}=     Get User Token      ${test_email}    ${TEST_USER_PASSWORD}

    # Try to create another user with non-admin token
    &{headers}=        Create Dictionary    Authorization=Bearer ${user_token}    Content-Type=application/json
    &{user_data}=      Create Dictionary    email=another-user@example.com    password=password123

    ${response}=       POST On Session    api    /api/users    json=${user_data}    headers=${headers}    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Cleanup
    Delete Test User    ${admin_token}    ${test_user}[user_id]

Update User Test
    [Documentation]    Test updating a user (admin only)
    [Tags]             users    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User    ${admin_token}    ${test_email}    ${TEST_USER_PASSWORD}

    # Update user to admin
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}    Content-Type=application/json
    &{update_data}=    Create Dictionary    email=${test_email}    password=${TEST_USER_PASSWORD}    is_superuser=${True}

    ${response}=       PUT On Session    api    /api/users/${user}[user_id]    json=${update_data}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${updated_user}=   Set Variable    ${response.json()}
    Should Be True     ${updated_user}[is_superuser]

    # Cleanup
    Delete Test User    ${admin_token}    ${user}[user_id]

Delete User Test
    [Documentation]    Test deleting a user (admin only)
    [Tags]             users    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User    ${admin_token}    ${test_email}    ${TEST_USER_PASSWORD}

    # Delete the user
    Delete Test User    ${admin_token}    ${user}[user_id]

    # Verify user is deleted by trying to login
    &{auth_data}=      Create Dictionary    username=${test_email}    password=${TEST_USER_PASSWORD}
    &{headers}=        Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=       POST On Session    api    /auth/jwt/login    data=${auth_data}    headers=${headers}    expected_status=400
    Should Be Equal As Integers    ${response.status_code}    400

