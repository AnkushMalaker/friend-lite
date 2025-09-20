*** Settings ***
Documentation    System and Admin API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Library          OperatingSystem
Resource         resources/setup_resources.robot
Resource         resources/auth_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get Auth Config Test
    [Documentation]    Test getting authentication configuration (public endpoint)
    [Tags]             auth    config    public    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /api/auth/config
    Should Be Equal As Integers    ${response.status_code}    200

    ${config}=      Set Variable    ${response.json()}
    # Auth config structure may vary, just verify it's valid JSON
    Should Be True    isinstance($config, dict)

Get Diarization Settings Test
    [Documentation]    Test getting diarization settings (admin only)
    [Tags]             system    diarization    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/diarization-settings    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${settings}=       Set Variable    ${response.json()}
    Should Be True     isinstance($settings, dict)

Save Diarization Settings Test
    [Documentation]    Test saving diarization settings (admin only)
    [Tags]             system    diarization    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}    Content-Type=application/json

    # First get current settings
    ${get_response}=   GET On Session    api    /api/diarization-settings    headers=${headers}
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${current_settings}=    Set Variable    ${get_response.json()}

    # Save the same settings (should succeed)
    ${response}=       POST On Session    api    /api/diarization-settings    json=${current_settings}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

Get Speaker Configuration Test
    [Documentation]    Test getting user's speaker configuration
    [Tags]             system    speakers    user    positive
    Setup Auth Session

    ${token}=          Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${token}

    ${response}=       GET On Session    api    /api/speaker-configuration    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${config}=         Set Variable    ${response.json()}
    Should Be True     isinstance($config, (dict, list))

Update Speaker Configuration Test
    [Documentation]    Test updating user's speaker configuration
    [Tags]             system    speakers    user    positive
    Setup Auth Session

    ${token}=          Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json

    # Update with empty speaker list
    ${speakers}=       Create List
    ${response}=       POST On Session    api    /api/speaker-configuration    json=${speakers}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

Get Enrolled Speakers Test
    [Documentation]    Test getting enrolled speakers from speaker recognition service
    [Tags]             system    speakers    service    positive
    Setup Auth Session

    ${token}=          Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${token}

    # This might fail if speaker service is not available
    ${response}=       GET On Session    api    /api/enrolled-speakers    headers=${headers}    expected_status=any
    Should Be True     ${response.status_code} in [200, 503]

    IF    ${response.status_code} == 200
        ${speakers}=   Set Variable    ${response.json()}
        Should Be True    isinstance($speakers, (dict, list))
    END

Get Speaker Service Status Test
    [Documentation]    Test checking speaker recognition service status (admin only)
    [Tags]             system    speakers    service    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/speaker-service-status    headers=${headers}    expected_status=any
    Should Be True     ${response.status_code} in [200, 503]

    ${status}=         Set Variable    ${response.json()}
    Should Be True     isinstance($status, dict)

Get Memory Config Raw Test
    [Documentation]    Test getting raw memory configuration (admin only)
    [Tags]             system    memory    config    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/admin/memory/config/raw    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    # Raw config should be text/yaml
    ${config}=         Set Variable    ${response.text}
    Should Not Be Empty    ${config}

Validate Memory Config Test
    [Documentation]    Test validating memory configuration YAML (admin only)
    [Tags]             system    memory    config    validation    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}    Content-Type=application/json

    # Test with valid YAML
    ${valid_yaml}=     Set Variable    memory_provider: "friend_lite"\nextraction:\n  enabled: true
    &{data}=           Create Dictionary    config_yaml=${valid_yaml}
    ${response}=       POST On Session    api    /api/admin/memory/config/validate    json=${data}    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    # Test with invalid YAML
    ${invalid_yaml}=   Set Variable    invalid: yaml: structure:
    &{data}=           Create Dictionary    config_yaml=${invalid_yaml}
    ${response}=       POST On Session    api    /api/admin/memory/config/validate    json=${data}    headers=${headers}    expected_status=any
    Should Be True     ${response.status_code} in [400, 422]

Reload Memory Config Test
    [Documentation]    Test reloading memory configuration (admin only)
    [Tags]             system    memory    config    reload    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       POST On Session    api    /api/admin/memory/config/reload    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

Delete All User Memories Test
    [Documentation]    Test deleting all memories for current user
    [Tags]             system    memory    delete    user    positive
    Setup Auth Session

    ${token}=          Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${token}

    ${response}=       DELETE On Session    api    /api/admin/memory/delete-all    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${result}=         Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${result}    message

List Processing Jobs Test
    [Documentation]    Test listing processing jobs (admin only)
    [Tags]             system    processing    jobs    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/process-audio-files/jobs    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${jobs}=           Set Variable    ${response.json()}
    Should Be True     isinstance($jobs, (dict, list))

Non-Admin Cannot Access Admin Endpoints Test
    [Documentation]    Test that non-admin users cannot access admin endpoints
    [Tags]             system    security    negative
    Setup Auth Session

    ${admin_token}=    Get Admin Token

    # Create a non-admin user
    ${test_user}=      Create Test User    ${admin_token}    test-user-${RANDOM_ID}@example.com    test-password-123
    ${user_token}=     Get User Token      test-user-${RANDOM_ID}@example.com    test-password-123

    &{headers}=        Create Dictionary    Authorization=Bearer ${user_token}

    # Test various admin endpoints
    ${endpoints}=      Create List
    ...                /api/diarization-settings
    ...                /api/speaker-service-status
    ...                /api/admin/memory/config/raw
    ...                /api/admin/memory/config/reload
    ...                /api/process-audio-files/jobs

    FOR    ${endpoint}    IN    @{endpoints}
        ${response}=   GET On Session    api    ${endpoint}    headers=${headers}    expected_status=403
        Should Be Equal As Integers    ${response.status_code}    403
    END

    # Cleanup
    Delete Test User    ${admin_token}    ${test_user}[id]

Unauthorized System Access Test
    [Documentation]    Test that system endpoints require authentication
    [Tags]             system    security    negative
    Setup Auth Session

    # Test endpoints that require authentication
    ${auth_endpoints}=    Create List
    ...                   /api/diarization-settings
    ...                   /api/speaker-configuration
    ...                   /api/enrolled-speakers
    ...                   /api/admin/memory/delete-all

    FOR    ${endpoint}    IN    @{auth_endpoints}
        ${response}=   GET On Session    api    ${endpoint}    expected_status=401
        Should Be Equal As Integers    ${response.status_code}    401
    END

Invalid System Operations Test
    [Documentation]    Test invalid operations on system endpoints
    [Tags]             system    negative    validation
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}    Content-Type=application/json

    # Test saving invalid diarization settings
    ${invalid_settings}=    Create Dictionary    invalid_key=invalid_value
    ${response}=             POST On Session    api    /api/diarization-settings    json=${invalid_settings}    headers=${headers}    expected_status=any
    Should Be True          ${response.status_code} in [400, 422]

    # Test updating memory config with invalid YAML
    ${invalid_yaml}=    Set Variable    {invalid yaml content
    &{data}=            Create Dictionary    config_yaml=${invalid_yaml}
    ${response}=        POST On Session    api    /api/admin/memory/config/raw    json=${data}    headers=${headers}    expected_status=any
    Should Be True      ${response.status_code} in [400, 422]

Memory Configuration Workflow Test
    [Documentation]    Test complete memory configuration workflow (admin only)
    [Tags]             system    memory    config    workflow    admin
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}    Content-Type=application/json

    # 1. Get current config
    ${get_response}=   GET On Session    api    /api/admin/memory/config/raw    headers=${headers}
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${original_config}=    Set Variable    ${get_response.text}

    # 2. Validate the current config
    &{validate_data}=  Create Dictionary    config_yaml=${original_config}
    ${validate_response}=    POST On Session    api    /api/admin/memory/config/validate    json=${validate_data}    headers=${headers}
    Should Be Equal As Integers    ${validate_response.status_code}    200

    # 3. Reload config (should succeed)
    ${reload_response}=    POST On Session    api    /api/admin/memory/config/reload    headers=${headers}
    Should Be Equal As Integers    ${reload_response.status_code}    200

Speaker Configuration Workflow Test
    [Documentation]    Test complete speaker configuration workflow
    [Tags]             system    speakers    workflow    user
    Setup Auth Session

    ${token}=          Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json

    # 1. Get current speaker configuration
    ${get_response}=   GET On Session    api    /api/speaker-configuration    headers=${headers}
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${current_config}=    Set Variable    ${get_response.json()}

    # 2. Update speaker configuration (with empty list)
    ${empty_speakers}=    Create List
    ${update_response}=   POST On Session    api    /api/speaker-configuration    json=${empty_speakers}    headers=${headers}
    Should Be Equal As Integers    ${update_response.status_code}    200

    # 3. Verify the update
    ${verify_response}=   GET On Session    api    /api/speaker-configuration    headers=${headers}
    Should Be Equal As Integers    ${verify_response.status_code}    200
    ${updated_config}=    Set Variable    ${verify_response.json()}

    # Should be empty list now
    ${length}=         Get Length    ${updated_config}
    Should Be Equal As Integers    ${length}    0

