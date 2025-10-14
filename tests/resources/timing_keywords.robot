*** Settings ***
Documentation    Timing and Performance Measurement Keywords
Library          DateTime
Library          Collections

*** Variables ***
&{TIMING_DATA}    # Global timing storage

*** Keywords ***
Start Timer
    [Documentation]    Start a timer for a specific operation
    [Arguments]    ${operation_name}

    ${start_time}=    Get Current Date    result_format=epoch
    Set To Dictionary    ${TIMING_DATA}    ${operation_name}_start    ${start_time}

    Log    Started timer for: ${operation_name}    DEBUG
    RETURN    ${start_time}

Stop Timer
    [Documentation]    Stop a timer and return the duration
    [Arguments]    ${operation_name}

    ${end_time}=    Get Current Date    result_format=epoch

    # Get start time
    ${start_key}=    Set Variable    ${operation_name}_start
    ${start_time}=   Get From Dictionary    ${TIMING_DATA}    ${start_key}

    # Calculate duration
    ${duration}=    Evaluate    ${end_time} - ${start_time}

    # Store results
    Set To Dictionary    ${TIMING_DATA}    ${operation_name}_end       ${end_time}
    Set To Dictionary    ${TIMING_DATA}    ${operation_name}_duration  ${duration}

    ${duration_formatted}=    Evaluate    f"{${duration}:.2f}"
    Log    ${operation_name} completed in ${duration_formatted} seconds    INFO
    RETURN    ${duration}

Get Timer Duration
    [Documentation]    Get duration for a completed timer
    [Arguments]    ${operation_name}

    ${duration_key}=    Set Variable    ${operation_name}_duration
    ${duration}=        Get From Dictionary    ${TIMING_DATA}    ${duration_key}
    RETURN    ${duration}

Log Timing Summary
    [Documentation]    Log summary of all timing measurements

    Log    === TIMING SUMMARY ===    INFO

    ${total_time}=    Set Variable    0

    FOR    ${key}    IN    @{TIMING_DATA.keys()}
        ${is_duration}=    Evaluate    "${key}".endswith("_duration")
        IF    ${is_duration}
            ${operation}=    Evaluate    "${key}".replace("_duration", "")
            ${duration}=     Get From Dictionary    ${TIMING_DATA}    ${key}
            ${total_time}=   Evaluate    ${total_time} + ${duration}
            ${duration_formatted}=    Evaluate    f"{${duration}:.2f}"
            Log    ${operation}: ${duration_formatted}s    INFO
        END
    END

    ${total_formatted}=    Evaluate    f"{${total_time}:.2f}"
    Log    Total measured time: ${total_formatted}s    INFO
    Log    === END TIMING SUMMARY ===    INFO

Reset Timers
    [Documentation]    Clear all timing data
    &{empty_dict}=    Create Dictionary
    Set Global Variable    &{TIMING_DATA}    &{empty_dict}
    Log    All timers reset    DEBUG

Time Operation
    [Documentation]    Time a single operation and return duration
    [Arguments]    ${operation_name}    ${keyword}    @{args}

    Start Timer    ${operation_name}
    ${result}=    Run Keyword    ${keyword}    @{args}
    ${duration}=    Stop Timer    ${operation_name}

    RETURN    ${result}    ${duration}

Check Performance Thresholds
    [Documentation]    Check if operations completed within expected thresholds
    [Arguments]    &{thresholds}

    ${violations}=    Create List

    FOR    ${operation}    ${threshold}    IN    &{thresholds}
        ${duration_key}=    Set Variable    ${operation}_duration
        ${duration}=        Get From Dictionary    ${TIMING_DATA}    ${duration_key}    default=0

        IF    ${duration} > ${threshold}
            ${duration_formatted}=    Evaluate    f"{${duration}:.2f}"
            Append To List    ${violations}    ${operation}: ${duration_formatted}s > ${threshold}s
            Log    Performance threshold exceeded: ${operation}    WARN
        ELSE
            Log    Performance threshold met: ${operation}    DEBUG
        END
    END

    ${violation_count}=    Get Length    ${violations}
    IF    ${violation_count} > 0
        ${violation_msg}=    Catenate    SEPARATOR=\n    @{violations}
        Log    Performance violations found:\n${violation_msg}    WARN
        RETURN    ${False}
    ELSE
        Log    All performance thresholds met    INFO
        RETURN    ${True}
    END

Create Performance Report
    [Documentation]    Create a detailed performance report
    [Arguments]    ${report_title}=Performance Report

    ${timestamp}=    Get Current Date    result_format=%Y-%m-%d %H:%M:%S

    Log    === ${report_title} (${timestamp}) ===    INFO

    # Calculate total test time
    ${total_time}=    Set Variable    0
    &{operation_times}=    Create Dictionary

    FOR    ${key}    IN    @{TIMING_DATA.keys()}
        ${is_duration}=    Evaluate    "${key}".endswith("_duration")
        IF    ${is_duration}
            ${operation}=    Evaluate    "${key}".replace("_duration", "")
            ${duration}=     Get From Dictionary    ${TIMING_DATA}    ${key}
            ${total_time}=   Evaluate    ${total_time} + ${duration}
            Set To Dictionary    ${operation_times}    ${operation}    ${duration}
        END
    END

    # Log operations sorted by duration
    ${operations}=    Get Dictionary Keys    ${operation_times}
    ${sorted_ops}=    Evaluate    sorted($operations, key=lambda x: $operation_times[x], reverse=True)

    Log    Operations by duration (slowest first):    INFO
    FOR    ${operation}    IN    @{sorted_ops}
        ${duration}=    Get From Dictionary    ${operation_times}    ${operation}
        ${percentage}=    Evaluate    (${duration} / ${total_time}) * 100 if ${total_time} > 0 else 0
        ${duration_formatted}=    Evaluate    f"{${duration}:.2f}"
        ${percentage_formatted}=    Evaluate    f"{${percentage}:.1f}"
        Log    ${operation}: ${duration_formatted}s (${percentage_formatted}%)    INFO
    END

    ${total_formatted}=    Evaluate    f"{${total_time}:.2f}"
    Log    Total execution time: ${total_formatted}s    INFO
    Log    === End ${report_title} ===    INFO

    RETURN    &{operation_times}

Benchmark Operation
    [Documentation]    Benchmark an operation multiple times
    [Arguments]    ${operation_name}    ${iterations}    ${keyword}    @{args}

    ${durations}=    Create List

    FOR    ${i}    IN RANGE    ${iterations}
        ${iteration_name}=    Set Variable    ${operation_name}_iteration_${i+1}
        Start Timer    ${iteration_name}
        Run Keyword    ${keyword}    @{args}
        ${duration}=    Stop Timer    ${iteration_name}
        Append To List    ${durations}    ${duration}
    END

    # Calculate statistics
    ${total}=    Evaluate    sum($durations)
    ${avg}=      Evaluate    ${total} / ${iterations}
    ${min_val}=  Evaluate    min($durations)
    ${max_val}=  Evaluate    max($durations)

    Log    Benchmark results for ${operation_name} (${iterations} iterations):    INFO
    ${avg_formatted}=     Evaluate    f"{${avg}:.3f}"
    ${min_formatted}=     Evaluate    f"{${min_val}:.3f}"
    ${max_formatted}=     Evaluate    f"{${max_val}:.3f}"
    Log    Average: ${avg_formatted}s, Min: ${min_formatted}s, Max: ${max_formatted}s    INFO

    &{stats}=    Create Dictionary
    ...    average=${avg}
    ...    minimum=${min_val}
    ...    maximum=${max_val}
    ...    total=${total}
    ...    iterations=${iterations}

    RETURN    &{stats}