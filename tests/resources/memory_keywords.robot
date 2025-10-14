*** Settings ***
Documentation    Memory Management Keywords
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py

*** Keywords ***

Get User Memories
    [Documentation]    Get memories for authenticated user
    [Arguments]    ${token}    ${limit}=50    ${user_id}=${None}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=${limit}

    IF    '${user_id}' != '${None}'
        Set To Dictionary    ${params}    user_id=${user_id}
    END

    ${response}=    GET On Session    api    /api/memories    headers=${headers}    params=${params}
    RETURN    ${response}

Get Memories With Transcripts
    [Documentation]    Get memories with their source transcripts
    [Arguments]    ${token}    ${limit}=50
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=${limit}

    ${response}=    GET On Session    api    /api/memories/with-transcripts    headers=${headers}    params=${params}
    RETURN    ${response}

Search Memories
    [Documentation]    Search memories by query
    [Arguments]    ${token}    ${query}    ${limit}=20    ${score_threshold}=0.0
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    query=${query}    limit=${limit}    score_threshold=${score_threshold}

    ${response}=    GET On Session    api    /api/memories/search    headers=${headers}    params=${params}
    RETURN    ${response}

Delete Memory
    [Documentation]    Delete a specific memory
    [Arguments]    ${token}    ${memory_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    DELETE On Session    api    /api/memories/${memory_id}    headers=${headers}
    RETURN    ${response}

Get Unfiltered Memories
    [Documentation]    Get all memories including fallback transcript memories
    [Arguments]    ${token}    ${limit}=50
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=${limit}

    ${response}=    GET On Session    api    /api/memories/unfiltered    headers=${headers}    params=${params}
    RETURN    ${response}

Get All Memories Admin
    [Documentation]    Get all memories across all users (admin only)
    [Arguments]    ${admin_token}    ${limit}=200
    &{headers}=    Create Dictionary    Authorization=Bearer ${admin_token}
    &{params}=     Create Dictionary    limit=${limit}

    ${response}=    GET On Session    api    /api/memories/admin    headers=${headers}    params=${params}
    RETURN    ${response}


Count User Memories
    [Documentation]    Count memories for a user
    [Arguments]    ${token}
    ${response}=    Get User Memories    ${token}    1000
    Should Be Equal As Integers    ${response.status_code}    200
    ${memories}=    Set Variable    ${response.json()}
    ${count}=       Get Length    ${memories}
    RETURN    ${count}