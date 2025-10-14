*** Settings ***
Documentation    System and Admin API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Library          OperatingSystem
Resource         ../resources/setup_resources.robot
Resource         ../resources/session_resources.robot
Resource         ../resources/user_resources.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get Diarization Settings Test
    [Documentation]    Test getting diarization settings (admin only)
    [Tags]             system    diarization    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    session
    ${response}=       GET On Session    session    /api/diarization-settings
    Should Be Equal As Integers    ${response.status_code}    200

    ${settings}=       Set Variable    ${response.json()}
    Should Be True     isinstance($settings, dict)

Save Diarization Settings Test
    [Documentation]    Test saving diarization settings (admin only)
    [Tags]             system    diarization    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # First get current settings
    ${get_response}=   GET On Session    admin_session   /api/diarization-settings
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${current_settings}=    Set Variable    ${get_response.json()}

    # Save the same settings (should succeed)
    ${response}=       POST On Session    admin_session    /api/diarization-settings    json=${current_settings}
    Should Be Equal As Integers    ${response.status_code}    200

Get Speaker Configuration Test
    [Documentation]    Test getting user's speaker configuration
    [Tags]             system    speakers    user    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/speaker-configuration
    Should Be Equal As Integers    ${response.status_code}    200

    ${config}=         Set Variable    ${response.json()}
    Should Be True     isinstance($config, dict)

    # Verify expected structure
    Dictionary Should Contain Key    ${config}    primary_speakers
    Dictionary Should Contain Key    ${config}    user_id
    Dictionary Should Contain Key    ${config}    status
    Should Be True     isinstance($config['primary_speakers'], list)

Update Speaker Configuration Test
    [Documentation]    Test updating user's speaker configuration
    [Tags]             system    speakers    user    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Update with empty speaker list
    ${speakers}=       Create List
    ${response}=       POST On Session    admin_session    /api/speaker-configuration    json=${speakers}
    Should Be Equal As Integers    ${response.status_code}    200

Get Enrolled Speakers Test
    [Documentation]    Test getting enrolled speakers from speaker recognition service
    [Tags]             system    speakers    service    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/enrolled-speakers
    Should Be Equal As Integers    ${response.status_code}    200

    ${response_data}=  Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${response_data}    service_available
    Dictionary Should Contain Key    ${response_data}    speakers

    # Check if speaker service is actually available
    IF    ${response_data}[service_available] == $False
        Skip    Speaker recognition service is not available or configured
    END

    # If service is available, verify speakers data
    Should Be True    isinstance($response_data[speakers], list)

Get Speaker Service Status Test
    [Documentation]    Test checking speaker recognition service status (admin only)
    [Tags]             system    speakers    service    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/speaker-service-status
    Should Be Equal As Integers    ${response.status_code}    200

    ${status}=         Set Variable    ${response.json()}
    Should Be True     isinstance($status, dict)

    # Verify expected keys are present
    Dictionary Should Contain Key    ${status}    service_available
    Dictionary Should Contain Key    ${status}    healthy
    Dictionary Should Contain Key    ${status}    status

    # Check if speaker service is actually available
    IF    ${status}[service_available] == $False
        Skip    Speaker recognition service is not available or configured
    END

Get Memory Config Raw Test
    [Documentation]    Test getting raw memory configuration (admin only)
    [Tags]             system    memory    config    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/admin/memory/config/raw
    Should Be Equal As Integers    ${response.status_code}    200

    # Raw config should be text/yaml
    ${config}=         Set Variable    ${response.text}
    Should Not Be Empty    ${config}

Validate Memory Config Test
    [Documentation]    Test validating memory configuration YAML (admin only)
    [Tags]             system    memory    config    validation    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test with valid YAML
    ${valid_yaml}=     Set Variable    memory_provider: "friend_lite"\nextraction:\n  enabled: true
    &{data}=           Create Dictionary    config_yaml=${valid_yaml}
    ${response}=       POST On Session    admin_session    /api/admin/memory/config/validate    json=${data}
    Should Be Equal As Integers    ${response.status_code}    200

    # Test with invalid YAML
    ${invalid_yaml}=   Set Variable    invalid: yaml: structure:
    &{data}=           Create Dictionary    config_yaml=${invalid_yaml}
    ${response}=       POST On Session    admin_session    /api/admin/memory/config/validate    json=${data}    expected_status=any
    Should Be True     ${response.status_code} in [400, 422]

Reload Memory Config Test
    [Documentation]    Test reloading memory configuration (admin only)
    [Tags]             system    memory    config    reload    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       POST On Session    admin_session    /api/admin/memory/config/reload
    Should Be Equal As Integers    ${response.status_code}    200

Delete All User Memories Test
    [Documentation]    Test deleting all memories for current user
    [Tags]             system    memory    delete    user    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       DELETE On Session    admin_session   /api/admin/memory/delete-all
    Should Be Equal As Integers    ${response.status_code}    200

    ${result}=         Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${result}    message

List Processing Jobs Test
    [Documentation]    Test listing processing jobs (admin only)
    [Tags]             system    processing    jobs    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/process-audio-files/jobs
    Should Be Equal As Integers    ${response.status_code}    200

    ${jobs}=           Set Variable    ${response.json()}
    Should Be True     isinstance($jobs, (dict, list))

Non-Admin Cannot Access Admin Endpoints Test
    [Documentation]    Test that non-admin users cannot access admin endpoints
    [Tags]             system    security    negative
    Get Anonymous Session    anon_session

    Create API Session    session

    # Create a non-admin user
    ${test_user}=      Create Test User    session   test-user-${RANDOM_ID}@example.com    test-password-123
    Create API Session    user_session    email=test-user-${RANDOM_ID}@example.com    password=test-password-123

    # Test various admin endpoints
    ${endpoints}=      Create List
    ...                /api/diarization-settings
    ...                /api/speaker-service-status
    ...                /api/admin/memory/config/raw
    ...                /api/admin/memory/config/reload
    ...                /api/process-audio-files/jobs

    FOR    ${endpoint}    IN    @{endpoints}
      ${response}=   GET On Session    user_session    ${endpoint}    expected_status=any
      Should Be True    ${response.status_code} in [405, 403]
    END

    # Cleanup
    Delete Test User    session    ${test_user}[user_id]

Unauthorized System Access Test
    [Documentation]    Test that system endpoints require authentication
    [Tags]             system    security    negative
    Get Anonymous Session    anon_session

    # Test endpoints that require authentication
    ${auth_endpoints}=    Create List
    ...                   /api/diarization-settings
    ...                   /api/speaker-configuration
    ...                   /api/enrolled-speakers
    ...                   /api/admin/memory/delete-all

    FOR    ${endpoint}    IN    @{auth_endpoints}
        ${response}=   GET On Session    anon_session    ${endpoint}    expected_status=any
        Should Be True    ${response.status_code} in [401, 405, 403]
    END

Invalid System Operations Test
    [Documentation]    Test invalid operations on system endpoints
    [Tags]             system    negative    validation
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test saving invalid diarization settings
    ${invalid_settings}=    Create Dictionary    invalid_key=invalid_value
    ${response}=             POST On Session    admin_session    /api/diarization-settings    json=${invalid_settings}    expected_status=any
    Should Be True          ${response.status_code} in [400, 422]

    # Test updating memory config with invalid YAML
    ${invalid_yaml}=    Set Variable    {invalid yaml content
    &{data}=            Create Dictionary    config_yaml=${invalid_yaml}
    ${response}=        POST On Session    admin_session    /api/admin/memory/config/raw    json=${data}    expected_status=any
    Should Be True      ${response.status_code} in [400, 422]

Memory Configuration Workflow Test
    [Documentation]    Test complete memory configuration workflow (admin only)
    [Tags]             system    memory    config    workflow    admin
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # 1. Get current config
    ${get_response}=   GET On Session    admin_session    /api/admin/memory/config/raw
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${original_config}=    Set Variable    ${get_response.text}

    # 2. Validate the current config
    &{validate_data}=  Create Dictionary    config_yaml=${original_config}
    ${validate_response}=    POST On Session    admin_session    /api/admin/memory/config/validate    json=${validate_data}
    Should Be Equal As Integers    ${validate_response.status_code}    200

    # 3. Reload config (should succeed)
    ${reload_response}=    POST On Session    admin_session    /api/admin/memory/config/reload
    Should Be Equal As Integers    ${reload_response.status_code}    200

Speaker Configuration Workflow Test
    [Documentation]    Test complete speaker configuration workflow
    [Tags]             system    speakers    workflow    user
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # 1. Get current speaker configuration
    ${get_response}=   GET On Session    admin_session    /api/speaker-configuration
    Should Be Equal As Integers    ${get_response.status_code}    200
    ${current_config}=    Set Variable    ${get_response.json()}

    # 2. Update speaker configuration (with empty list)
    ${empty_speakers}=    Create List
    ${update_response}=   POST On Session    admin_session    /api/speaker-configuration    json=${empty_speakers}
    Should Be Equal As Integers    ${update_response.status_code}    200

    # 3. Verify the update
    ${verify_response}=   GET On Session    admin_session   /api/speaker-configuration
    Should Be Equal As Integers    ${verify_response.status_code}    200
    ${updated_config}=    Set Variable    ${verify_response.json()}

    # Verify response structure
    Dictionary Should Contain Key    ${updated_config}    primary_speakers
    Dictionary Should Contain Key    ${updated_config}    user_id
    Dictionary Should Contain Key    ${updated_config}    status

    # Should be empty list now
    ${speakers_list}=  Set Variable    ${updated_config}[primary_speakers]
    ${length}=         Get Length    ${speakers_list}
    Should Be Equal As Integers    ${length}    0

