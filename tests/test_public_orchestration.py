import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "run_public_gpu_validation_queue.sh"
PGE_LAUNCHER = ROOT / "scripts" / "run_public_pge_after_horizontal.sh"
PGE_RUNNER = ROOT / "scripts" / "run_public_pge_validation.sh"
GLIREL_T0_LAUNCHER = ROOT / "scripts" / "run_gliner_glirel_t0_after_spert.sh"
GLIREL_CANARY = ROOT / "scripts" / "check_glirel_compatibility.py"
GLIREL_CALIBRATION = ROOT / "scripts" / "calibrate_glirel_train.py"
GLIREL_CALIBRATED_LAUNCHER = (
    ROOT / "scripts" / "run_gliner_glirel_calibrated_after_t0.sh"
)
GLIREL_CALIBRATED_RUNNER = (
    ROOT / "scripts" / "run_gliner_glirel_calibrated_validation.sh"
)
POST_PGE_LAUNCHER = ROOT / "scripts" / "launch_public_post_pge_bootstrap.sh"
VALIDATION_AUDIT = ROOT / "scripts" / "audit_public_validation_results.py"
VALIDATION_AUDIT_LAUNCHER = (
    ROOT / "scripts" / "run_public_validation_audit_after_queue.sh"
)
WATCHDOG = ROOT / "scripts" / "monitor_public_experiments.sh"
STATUS = ROOT / "scripts" / "public_experiment_status.sh"


def test_orchestration_shell_scripts_have_valid_syntax():
    for script in (
        QUEUE,
        PGE_LAUNCHER,
        PGE_RUNNER,
        GLIREL_T0_LAUNCHER,
        GLIREL_CALIBRATED_LAUNCHER,
        GLIREL_CALIBRATED_RUNNER,
        POST_PGE_LAUNCHER,
        VALIDATION_AUDIT_LAUNCHER,
        WATCHDOG,
        STATUS,
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_watchdog_accepts_each_declared_terminal_queue_state(tmp_path):
    source = WATCHDOG.read_text(encoding="utf-8")
    start = source.index("marker_status() {")
    end = source.index("\n\nlaunch_managed_job()", start)
    functions = source[start:end]
    marker = tmp_path / "status.json"

    for state in ("complete", "complete_with_terminal_failures"):
        marker.write_text(json.dumps({"status": state}), encoding="utf-8")
        subprocess.run(
            [
                "bash",
                "-c",
                functions + '\nmarker_complete "$1" "$2"',
                "watchdog-test",
                str(marker),
                "complete|complete_with_terminal_failures",
            ],
            check=True,
        )


def test_queue_propagates_qwen_terminal_failures_and_pge_accepts_terminal_queue():
    queue = QUEUE.read_text(encoding="utf-8")
    launcher = PGE_LAUNCHER.read_text(encoding="utf-8")

    assert "qwen_remaining_failed_jobs" in queue
    assert "write_status complete_with_terminal_failures" in queue
    assert "horizontal_is_terminal" in launcher
    assert 'state" == "complete_with_terminal_failures"' in launcher


def test_pge_watchdog_tracks_waiting_training_and_runner_progress():
    watchdog = WATCHDOG.read_text(encoding="utf-8")

    assert "outputs/public_pge_validation_seed42/launcher_status.json" in watchdog
    assert "outputs/public_pge_validation_seed42/status.json" in watchdog
    assert "outputs/public_pge_validation_seed42/runner.log" in watchdog
    assert "public-spert-fresh-validation" in watchdog
    assert "outputs/public_horizontal_validation/spert_fresh/launcher_status.json" in watchdog
    assert "outputs/public_horizontal_validation/spert_fresh/runner.log" in watchdog


def test_watchdog_checks_gpu_liveness_by_owning_cgroup_and_not_global_usage():
    watchdog = WATCHDOG.read_text(encoding="utf-8")

    assert "--query-compute-apps=pid" in watchdog
    assert "-p ControlGroup --value" in watchdog
    assert '"/sys/fs/cgroup${control_group}/cgroup.procs"' in watchdog
    assert "runtime_cuda_pids" in watchdog
    assert "gpu_liveness job=%s" in watchdog
    assert "interval_minutes=20" in watchdog
    assert "utilization.gpu" not in watchdog.split("check_gpu_liveness()", 1)[1].split(
        "glirel_t0_blocked()", 1
    )[0]


def test_watchdog_orchestrates_formal_qwen_after_completed_release():
    watchdog = WATCHDOG.read_text(encoding="utf-8")
    formal_root = "outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot"

    phase_function = watchdog.split("gpu_phase_is_expected() {", 1)[1].split(
        "clear_gpu_suspicion()", 1
    )[0]
    formal_qwen_phase = phase_function.split("    formal_qwen)", 1)[1].split(
        "      ;;", 1
    )[0]
    assert f'{formal_root}/status.json' in formal_qwen_phase
    assert '"$state" == running && "$stage" == test_inference' in formal_qwen_phase

    launch_function = watchdog.split("launch_managed_job() {", 1)[1].split(
        "repair_managed_job()", 1
    )[0]
    formal_qwen_launch = launch_function.split("    formal_qwen)", 1)[1].split(
        "      ;;", 1
    )[0]
    assert "--unit=public-formal-qwen-zeroshot" in formal_qwen_launch
    assert f"mkdir -p {formal_root}" in formal_qwen_launch
    assert f"exec >{formal_root}/runner.log 2>&1" in formal_qwen_launch
    assert "exec scripts/run_qwen_zeroshot_formal_test.sh" in formal_qwen_launch

    gate_block = watchdog.split('formal_gate_review="', 1)[1].split(
        "# Query once per watchdog cycle", 1
    )[0]
    release_guard = (
        "if marker_complete outputs/public_formal_matrix/release_status.json "
        "complete; then"
    )
    assert gate_block.index(release_guard) < gate_block.index(
        "formal_qwen public-formal-qwen-zeroshot.service"
    )
    assert "managed_job=formal_qwen status=waiting_for_formal_release" in gate_block

    assert "formal_qwen_expected=no" in watchdog
    assert (
        "formal_qwen public-formal-qwen-zeroshot.service \\\n"
        f"  {formal_root}/status.json complete"
    ) in watchdog
    assert (
        "public-formal-qwen-zeroshot.service \\\n"
        f"  {formal_root}/status.json complete"
    ) in watchdog
    assert '"$cmdline" == *"run_qlora_inference.py"*' in watchdog


def test_watchdog_gpu_recovery_is_phase_gated_and_protects_cpu_work():
    watchdog = WATCHDOG.read_text(encoding="utf-8")

    # Generic lifecycle repair must not restart active waiting/CPU jobs merely
    # because an old progress file is quiet.
    repair = watchdog.split("repair_managed_job()", 1)[1].split(
        "repair_download_lane()", 1
    )[0]
    assert "systemctl --user restart" not in repair.split(
        'if [[ "$state" == "active"', 1
    )[1].split("return 0", 1)[0]
    assert "reason=stalled" not in repair

    for protected_phase in (
        "--preflight-only",
        "preflight_",
        "prepare_",
        "evaluate_",
        "convert_",
        "merge_",
        "materialize_",
        "bootstrap_",
        "calibrate_",
        "audit_",
    ):
        assert protected_phase in watchdog

    assert '"$pge_stage" == training' in watchdog
    assert '"$pge_stage" == validation_inference' in watchdog
    assert '"$spert_phase" == running_spert_fresh' in watchdog
    assert '"$t0_state" == running' in watchdog
    assert '"$calibrated_execution" == fresh_inference' in watchdog
    assert "GPU_LIVENESS_GRACE_SECONDS" in watchdog
    assert "GPU_RESTART_COOLDOWN_SECONDS" in watchdog
    assert "action=none_after_recheck" in watchdog


def _run_liveness_harness(tmp_path, overrides, calls=1):
    source = WATCHDOG.read_text(encoding="utf-8")
    start = source.index("clear_gpu_suspicion() {")
    end = source.index("\n\nglirel_t0_blocked()", start)
    function = source[start:end]
    program = "\n".join(
        (
            "set -uo pipefail",
            'timestamp="test-time"',
            'state_root="$1"',
            'log="$state_root/watchdog.log"',
            'restart_flag="$state_root/restarted"',
            "gpu_liveness_grace_seconds=10",
            "gpu_restart_cooldown_seconds=3600",
            "gpu_query_status=ok",
            'gpu_compute_pids=""',
            "systemctl() {",
            '  case " $* " in',
            '    *" is-active "*) printf "active\\n" ;;',
            '    *" -p SubState --value "*) printf "running\\n" ;;',
            '    *" restart "*) printf "restart\\n" >> "$restart_flag" ;;',
            "  esac",
            "}",
            'latest_progress_mtime() { printf "1\\n"; }',
            'unit_active_age_seconds() { printf "9999\\n"; }',
            function,
            overrides,
            *(
                'check_gpu_liveness demo demo.service marker.json complete yes running "$state_root"'
                for _ in range(calls)
            ),
        )
    )
    return subprocess.run(
        ["bash", "-c", program, "watchdog-liveness-test", str(tmp_path)],
        check=False,
        text=True,
        capture_output=True,
    )


def test_watchdog_aborts_recovery_if_second_gpu_query_fails(tmp_path):
    result = _run_liveness_harness(
        tmp_path,
        "\n".join(
            (
                "marker_complete() { return 1; }",
                "gpu_phase_is_expected() { return 0; }",
                "read_unit_runtime() {",
                '  runtime_cuda_pids=""; runtime_gpu_worker_pids=""',
                '  runtime_cpu_phase_pids=""; runtime_d_state_pids=""',
                "}",
                'refresh_gpu_snapshot() { gpu_query_status="failed"; gpu_compute_pids=""; }',
            )
        ),
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "restarted").exists()
    assert "query=failed action=none_after_recheck" in (
        tmp_path / "watchdog.log"
    ).read_text(encoding="utf-8")


def test_watchdog_aborts_recovery_if_job_completes_during_recheck(tmp_path):
    result = _run_liveness_harness(
        tmp_path,
        "\n".join(
            (
                "marker_checks=0",
                "marker_complete() { marker_checks=$((marker_checks + 1)); (( marker_checks >= 2 )); }",
                "gpu_phase_is_expected() { return 0; }",
                "read_unit_runtime() {",
                '  runtime_cuda_pids=""; runtime_gpu_worker_pids=""',
                '  runtime_cpu_phase_pids=""; runtime_d_state_pids=""',
                "}",
                'refresh_gpu_snapshot() { gpu_query_status="ok"; gpu_compute_pids=""; }',
            )
        ),
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "restarted").exists()
    assert "reason=completed_during_recheck action=none" in (
        tmp_path / "watchdog.log"
    ).read_text(encoding="utf-8")


def test_watchdog_observes_same_pre_cuda_worker_twice_before_recovery(tmp_path):
    result = _run_liveness_harness(
        tmp_path,
        "\n".join(
            (
                "marker_complete() { return 1; }",
                "gpu_phase_is_expected() { return 0; }",
                "read_unit_runtime() {",
                '  runtime_cuda_pids=""; runtime_gpu_worker_pids="4321"',
                '  runtime_cpu_phase_pids=""; runtime_d_state_pids=""',
                "}",
                'refresh_gpu_snapshot() { gpu_query_status="ok"; gpu_compute_pids=""; }',
            )
        ),
        calls=2,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "restarted").read_text(encoding="utf-8").splitlines() == [
        "restart"
    ]
    log = (tmp_path / "watchdog.log").read_text(encoding="utf-8")
    assert "strike=1/2 action=observe" in log
    assert "action=restarted" in log


def test_watchdog_healthy_observation_clears_old_gpu_suspicion(tmp_path):
    result = _run_liveness_harness(
        tmp_path,
        "\n".join(
            (
                "marker_complete() { return 1; }",
                "gpu_phase_is_expected() { return 0; }",
                "runtime_reads=0",
                "read_unit_runtime() {",
                "  runtime_reads=$((runtime_reads + 1))",
                '  runtime_gpu_worker_pids="4321"',
                '  runtime_cpu_phase_pids=""; runtime_d_state_pids=""',
                '  if (( runtime_reads == 3 )); then runtime_cuda_pids="4321"; else runtime_cuda_pids=""; fi',
                "}",
                'refresh_gpu_snapshot() { gpu_query_status="ok"; gpu_compute_pids=""; }',
            )
        ),
        calls=3,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "restarted").exists()
    log = (tmp_path / "watchdog.log").read_text(encoding="utf-8")
    assert log.count("strike=1/2 action=observe") == 2


def test_glirel_t0_is_an_independent_resumable_post_spert_queue():
    launcher = GLIREL_T0_LAUNCHER.read_text(encoding="utf-8")
    watchdog = WATCHDOG.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")

    assert 'run_root="outputs/public_horizontal_validation/gliner_glirel_t0"' in launcher
    assert 'spert_status="outputs/public_horizontal_validation/spert_fresh/status.json"' in launcher
    assert '"complete"' in launcher
    assert 'requested_workers="${GLINER_GLIREL_T0_WORKERS:-2}"' in launcher
    assert "ENTITY_THRESHOLD=0.5" in launcher
    assert "RELATION_THRESHOLD=0.0" in launcher
    assert 'GLINER_GLIREL_WORKERS="$requested_workers"' in launcher
    assert 'preserves": "outputs/public_horizontal_validation/gliner_glirel"' in launcher
    assert "scripts/check_glirel_compatibility.py" in launcher
    assert "blocked_incompatible_runtime" in launcher
    assert "CUDA_VISIBLE_DEVICES=''" in launcher

    assert "public-gliner-glirel-t0-validation.service" in watchdog
    assert "glirel_t0_blocked" in watchdog
    assert "status=blocked_incompatible_runtime action=none" in watchdog
    for progress_path in (
        "outputs/public_horizontal_validation/gliner_glirel_t0/launcher_status.json",
        "outputs/public_horizontal_validation/gliner_glirel_t0/status.json",
        "outputs/public_horizontal_validation/gliner_glirel_t0/runner.log",
    ):
        assert progress_path in watchdog
    assert "relation-threshold=0 sensitivity" in status
    assert 't0_root="outputs/public_horizontal_validation/gliner_glirel_t0"' in status
    assert "GLiREL train-only calibration" in status

    assert '"public-glirel-calibration-${dataset}.service"' in watchdog
    assert "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}" in watchdog


def test_calibrated_glirel_queue_reuses_identical_t0_arms_without_selection_leakage():
    launcher = GLIREL_CALIBRATED_LAUNCHER.read_text(encoding="utf-8")
    runner = GLIREL_CALIBRATED_RUNNER.read_text(encoding="utf-8")
    watchdog = WATCHDOG.read_text(encoding="utf-8")

    assert "waiting_for_t0_and_calibration" in launcher
    assert '"validation_gold_used_for_selection": False' in launcher
    assert '"test_gold_used_for_selection": False' in launcher
    assert "train_inner_calibration" in launcher
    assert "reuse_canonical_t0_predictions" in launcher
    assert "existing calibrated protocol differs" in launcher

    assert "PUBLIC_GPU_LOCK_FILE" in runner
    assert "flock -n 8" in runner
    assert "cp --reflink=auto" in runner
    assert "sha256sum" in runner
    assert "--relation-label-mode" in runner
    assert "--relation-threshold" in runner
    assert "--include-missing-as-empty" in runner

    assert "public-gliner-glirel-calibrated-validation.service" in watchdog
    assert "gliner_glirel_calibrated/launcher_status.json" in watchdog
    assert "gliner_glirel_calibrated/status.json" in watchdog


def test_glirel_compatibility_gate_uses_the_frozen_large_v0_runtime_contract():
    canary = GLIREL_CANARY.read_text(encoding="utf-8")

    assert '"followed by"' in canary
    assert "0.0028011202812194824" in canary
    assert "CANARY_NER" in canary
    assert "threshold=0.0" in canary
    assert "top_k=1" in canary
    assert 'default=1e-6' in canary
    assert "stale_beta_expected_output" in canary
    assert "semantic_accuracy_claim" in canary
    assert '"runtime_compatible": compatible' in canary
    assert '"status": "passed" if compatible else "failed"' in canary


def test_validation_audit_is_closed_validation_only_cpu_orchestration():
    worker = VALIDATION_AUDIT.read_text(encoding="utf-8")
    launcher = VALIDATION_AUDIT_LAUNCHER.read_text(encoding="utf-8")
    watchdog = WATCHDOG.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")

    assert 'SCHEMA_VERSION = "public-validation-audit-v1"' in worker
    assert '"closed_explicit_paths_no_discovery"' in worker
    assert '"test_namespace_status": TEST_ACCESS_SEAL' in worker
    assert '"formal_test_read": False' in worker
    assert "rglob(" not in worker
    assert "os.walk(" not in worker
    assert "summary.json" in worker
    assert "summary.tsv" in worker
    assert "input_manifest.json" in worker
    assert "terminal_failure_accounting" in worker
    assert "uncalibrated_diagnostic" in worker

    for marker in (
        "outputs/public_full_stage1/validation_analysis/status.json",
        "outputs/public_horizontal_validation/qwen3_4b_zero_shot/retry_status.json",
        "outputs/public_pge_validation_seed42/status.json",
        "outputs/public_horizontal_validation/spert_fresh/status.json",
        "outputs/public_horizontal_validation/gliner_glirel_t0/status.json",
        "outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json",
        "outputs/public_horizontal_validation/gliner_entity_only/status.json",
        "outputs/public_post_pge_validation_seed42/status.json",
    ):
        assert marker in launcher
    assert 'export CUDA_VISIBLE_DEVICES=""' in launcher
    assert "PUBLIC_GPU_LOCK_FILE" not in launcher

    assert "public-validation-audit.service" in watchdog
    assert "run_public_validation_audit_after_queue.sh" in watchdog
    assert "outputs/public_validation_audit/launcher_status.json" in watchdog
    assert "unified public validation audit" in status
    assert "test_namespace" in status


def test_watchdog_can_recover_entity_only_and_tracks_post_pge_without_gpu():
    watchdog = WATCHDOG.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")

    assert "public-gliner-entity-only-validation.service" in watchdog
    assert "derive_gliner_entity_only_validation.py --overwrite" in watchdog
    assert "waiting_for_gliner_glirel" in watchdog
    assert "public-post-pge-bootstrap.service" in watchdog
    assert "launch_public_post_pge_bootstrap.sh" in watchdog
    assert "formal GLiNER entity-only validation" in status
    assert "post-PGE validation bootstrap" in status


def test_pge_runner_enforces_exclusive_gpu_and_hashes_method_artifacts():
    runner = PGE_RUNNER.read_text(encoding="utf-8")

    assert "PUBLIC_GPU_LOCK_FILE" in runner
    assert "flock -n 8" in runner
    assert 'write_status waiting_for_gpu' in runner
    for artifact in (
        "training_metrics.json",
        "validation_complete.jsonl",
        "validation_expanded.jsonl",
        "evge_validation.jsonl",
        "cfe_validation.jsonl",
        "pge_validation.jsonl",
        "implementation_files",
        "soe_lineage",
        "missing_predictions.json",
        "expected_validation_jobs",
        "baseline_expanded.jsonl",
        "generator_coverage",
        "failures_materialized_as_empty",
    ):
        assert artifact in runner


def test_public_status_deduplicates_qwen_retry_log_rows_by_job_id():
    status = STATUS.read_text(encoding="utf-8")

    assert "reduce .[] as $row" in status
    assert 'success=%4d failed=%3d' in status
    assert "retry_status=" in status
