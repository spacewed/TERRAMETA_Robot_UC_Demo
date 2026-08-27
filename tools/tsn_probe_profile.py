#!/usr/bin/env python3
"""Enable or restore a TSN-like low-jitter profile for the UDP probe.

This tool intentionally targets the small latency-probe packets first.  It
marks UDP probe traffic, maps it to a high-priority transmit class, reduces NIC
interrupt moderation, and records enough state to restore the changed knobs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PORT = 10000
DEFAULT_DSCP = 46
DEFAULT_PRIORITY = 6
COMMENT_PREFIX = "terrameta-tsn-probe"
STATE_DIR = Path(os.environ.get("TM_TSN_STATE_DIR", "/var/tmp/terrameta_tsn"))
SYSCTL_TUNABLES = {
    "net.core.busy_read": "50",
    "net.core.busy_poll": "50",
    "kernel.timer_migration": "0",
    "kernel.sched_rt_runtime_us": "-1",
}
PTP_PRIVATE_FLAGS = {
    "tx_port_ts": "on",
}


class CommandError(RuntimeError):
    pass


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    dry_run: bool = False,
    capture: bool = False,
    warn: bool = False,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print(f"+ {quote_cmd(cmd)}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0 and check:
        stderr = (result.stderr or "").strip()
        message = f"command failed ({result.returncode}): {quote_cmd(cmd)}"
        if stderr:
            message = f"{message}\n{stderr}"
        if warn:
            print(f"warning: {message}", file=sys.stderr)
            return result
        raise CommandError(message)
    return result


def output(cmd: list[str]) -> str:
    result = run(cmd, check=False, capture=True)
    if result.returncode != 0:
        return (result.stderr or "").strip()
    return result.stdout or ""


def require_root_for_mutation(args: argparse.Namespace) -> None:
    if getattr(args, "dry_run", False):
        return
    if os.geteuid() != 0:
        raise CommandError("apply/restore requires root; rerun with sudo")


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def iface_exists(iface: str) -> bool:
    return Path("/sys/class/net", iface).exists()


def state_path(iface: str) -> Path:
    safe_iface = re.sub(r"[^A-Za-z0-9_.-]", "_", iface)
    return STATE_DIR / f"{safe_iface}.json"


def safe_iface_name(iface: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", iface)


def pid_path(iface: str, name: str) -> Path:
    return STATE_DIR / f"{safe_iface_name(iface)}_{name}.pid"


def log_path(iface: str, name: str) -> Path:
    return STATE_DIR / f"{safe_iface_name(iface)}_{name}.log"


def get_tx_queue_count(iface: str) -> int:
    queue_dir = Path("/sys/class/net", iface, "queues")
    return len(tuple(queue_dir.glob("tx-*")))


def parse_on_off(value: str) -> str | None:
    value = value.strip().lower()
    if value in {"on", "off"}:
        return value
    return None


def parse_coalesce(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        adaptive = re.match(r"Adaptive RX:\s+(\S+)\s+TX:\s+(\S+)", line)
        if adaptive:
            values["adaptive-rx"] = adaptive.group(1)
            values["adaptive-tx"] = adaptive.group(2)
            continue
        match = re.match(r"(rx-usecs|rx-frames|tx-usecs|tx-frames):\s+(\S+)", line)
        if match and match.group(2) != "n/a":
            values[match.group(1)] = match.group(2)
    return values


def parse_pause(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    keys = {
        "Autonegotiate": "autoneg",
        "RX": "rx",
        "TX": "tx",
    }
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        option = keys.get(key)
        if option and parse_on_off(value):
            values[option] = value.lower()
    return values


def parse_private_flags(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        match = re.match(r"([A-Za-z0-9_.-]+)\s*:\s*(on|off)$", line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_mlnx_qos(raw: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    trust = re.search(r"Priority trust state:\s*(\S+)", raw)
    if trust:
        values["trust"] = trust.group(1)

    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("enabled") and index > 0:
            numbers = re.findall(r"\b[01]\b", line)
            if len(numbers) >= 8:
                values["pfc"] = ",".join(numbers[:8])
                break

    tc_to_priority: dict[int, list[int]] = {}
    tsa_by_tc: dict[int, str] = {}
    current_tc: int | None = None
    for line in lines:
        tc_match = re.match(r"tc:\s*(\d+).*tsa:\s*(\S+)", line.strip())
        if tc_match:
            current_tc = int(tc_match.group(1))
            tsa_by_tc[current_tc] = tc_match.group(2).rstrip(",")
            continue
        prio_match = re.match(r"priority:\s*(.*)", line.strip())
        if prio_match and current_tc is not None:
            priorities = [int(item) for item in re.findall(r"\d+", prio_match.group(1))]
            tc_to_priority[current_tc] = priorities

    if tsa_by_tc:
        values["tsa"] = ",".join(tsa_by_tc.get(tc, "vendor") for tc in range(8))
    if tc_to_priority:
        prio_to_tc = [str(priority) for priority in range(8)]
        for tc, priorities in tc_to_priority.items():
            for priority in priorities:
                if 0 <= priority < 8:
                    prio_to_tc[priority] = str(tc)
        values["prio_tc"] = ",".join(prio_to_tc)

    return values


def snapshot_state(iface: str, profile: str) -> dict[str, Any]:
    coalesce_raw = output(["ethtool", "-c", iface]) if command_exists("ethtool") else ""
    pause_raw = output(["ethtool", "-a", iface]) if command_exists("ethtool") else ""
    private_flags_raw = (
        output(["ethtool", "--show-priv-flags", iface]) if command_exists("ethtool") else ""
    )
    mlnx_raw = output(["mlnx_qos", "-i", iface]) if command_exists("mlnx_qos") else ""
    return {
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "iface": iface,
        "profile": profile,
        "raw": {
            "qdisc": output(["tc", "qdisc", "show", "dev", iface]) if command_exists("tc") else "",
            "egress_filters": output(["tc", "filter", "show", "dev", iface, "egress"])
            if command_exists("tc")
            else "",
            "coalesce": coalesce_raw,
            "pause": pause_raw,
            "private_flags": private_flags_raw,
            "mlnx_qos": mlnx_raw,
        },
        "parsed": {
            "coalesce": parse_coalesce(coalesce_raw),
            "pause": parse_pause(pause_raw),
            "private_flags": parse_private_flags(private_flags_raw),
            "mlnx_qos": parse_mlnx_qos(mlnx_raw),
            "sysctl": read_sysctls(),
        },
    }


def save_snapshot(iface: str, profile: str, *, replace: bool, dry_run: bool) -> None:
    path = state_path(iface)
    if path.exists() and not replace:
        if not dry_run:
            state = load_snapshot(iface) or {}
            parsed = state.setdefault("parsed", {})
            if "sysctl" not in parsed:
                parsed["sysctl"] = read_sysctls()
                print(f"updated restore point with sysctl baseline: {path}")
            if "private_flags" not in parsed:
                parsed["private_flags"] = read_private_flags(iface)
                print(f"updated restore point with private-flag baseline: {path}")
            path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        print(f"state already exists at {path}; preserving original restore point")
        return
    state = snapshot_state(iface, profile)
    if dry_run:
        print(f"+ write state snapshot {path}")
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved restore point: {path}")


def load_snapshot(iface: str) -> dict[str, Any] | None:
    path = state_path(iface)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_snapshot(iface: str, *, dry_run: bool) -> None:
    path = state_path(iface)
    if not path.exists():
        return
    if dry_run:
        print(f"+ remove state snapshot {path}")
        return
    path.unlink()


def iptables_rule_spec(port: int, direction: str, dscp: int) -> list[str]:
    port_flag = "--dport" if direction == "dst" else "--sport"
    return [
        "-p",
        "udp",
        port_flag,
        str(port),
        "-m",
        "comment",
        "--comment",
        f"{COMMENT_PREFIX}-{direction}-{port}",
        "-j",
        "DSCP",
        "--set-dscp",
        str(dscp),
    ]


def iptables_has_rule(spec: list[str]) -> bool:
    result = run(["iptables", "-t", "mangle", "-C", "OUTPUT", *spec], check=False, capture=True)
    return result.returncode == 0


def add_iptables_rule(port: int, direction: str, dscp: int, *, dry_run: bool) -> None:
    if not command_exists("iptables"):
        print("warning: iptables not found; skipping DSCP marking", file=sys.stderr)
        return
    spec = iptables_rule_spec(port, direction, dscp)
    if not dry_run and iptables_has_rule(spec):
        return
    run(["iptables", "-t", "mangle", "-A", "OUTPUT", *spec], dry_run=dry_run)


def delete_iptables_rule(port: int, direction: str, dscp: int, *, dry_run: bool) -> None:
    if not command_exists("iptables"):
        return
    spec = iptables_rule_spec(port, direction, dscp)
    while True:
        result = run(
            ["iptables", "-t", "mangle", "-D", "OUTPUT", *spec],
            check=False,
            dry_run=dry_run,
            capture=True,
        )
        if dry_run or result.returncode != 0:
            return


def apply_coalesce(iface: str, *, dry_run: bool) -> None:
    if not command_exists("ethtool"):
        print("warning: ethtool not found; skipping coalescing", file=sys.stderr)
        return
    commands = [
        ["ethtool", "-C", iface, "adaptive-rx", "off", "adaptive-tx", "off"],
        ["ethtool", "-C", iface, "rx-usecs", "0", "tx-usecs", "0"],
        ["ethtool", "-C", iface, "rx-frames", "1", "tx-frames", "1"],
    ]
    for cmd in commands:
        run(cmd, check=False, dry_run=dry_run, warn=True)


def restore_coalesce(iface: str, state: dict[str, Any] | None, *, dry_run: bool) -> None:
    if not command_exists("ethtool") or state is None:
        return
    values = state.get("parsed", {}).get("coalesce", {})
    options: list[str] = []
    for key in ("adaptive-rx", "adaptive-tx", "rx-usecs", "tx-usecs", "rx-frames", "tx-frames"):
        value = values.get(key)
        if value:
            options.extend([key, str(value)])
    if options:
        run(["ethtool", "-C", iface, *options], check=False, dry_run=dry_run, warn=True)


def apply_pause(iface: str, *, dry_run: bool, keep_pause: bool) -> None:
    if keep_pause or not command_exists("ethtool"):
        return
    run(
        ["ethtool", "-A", iface, "autoneg", "off", "rx", "off", "tx", "off"],
        check=False,
        dry_run=dry_run,
        warn=True,
    )


def restore_pause(iface: str, state: dict[str, Any] | None, *, dry_run: bool) -> None:
    if not command_exists("ethtool") or state is None:
        return
    values = state.get("parsed", {}).get("pause", {})
    options: list[str] = []
    for key in ("autoneg", "rx", "tx"):
        value = values.get(key)
        if value:
            options.extend([key, value])
    if options:
        run(["ethtool", "-A", iface, *options], check=False, dry_run=dry_run, warn=True)


def read_private_flags(iface: str) -> dict[str, str]:
    if not command_exists("ethtool"):
        return {}
    return parse_private_flags(output(["ethtool", "--show-priv-flags", iface]))


def apply_private_flags(iface: str, desired: dict[str, str], *, dry_run: bool) -> None:
    if not command_exists("ethtool"):
        print("warning: ethtool not found; skipping private flags", file=sys.stderr)
        return
    current = read_private_flags(iface)
    for flag, value in desired.items():
        if flag not in current:
            print(f"warning: private flag {flag} not supported on {iface}; skipping", file=sys.stderr)
            continue
        run(
            ["ethtool", "--set-priv-flags", iface, flag, value],
            check=False,
            dry_run=dry_run,
            warn=True,
        )


def restore_private_flags(iface: str, state: dict[str, Any] | None, *, dry_run: bool) -> None:
    if not command_exists("ethtool") or state is None:
        return
    saved = state.get("parsed", {}).get("private_flags", {})
    if not saved:
        return
    current = read_private_flags(iface)
    for flag in PTP_PRIVATE_FLAGS:
        value = saved.get(flag)
        if not value:
            continue
        if flag not in current:
            print(f"warning: private flag {flag} not supported on {iface}; skipping restore", file=sys.stderr)
            continue
        run(
            ["ethtool", "--set-priv-flags", iface, flag, value],
            check=False,
            dry_run=dry_run,
            warn=True,
        )


def apply_mlnx_qos(iface: str, dscp: int, priority: int, *, dry_run: bool) -> None:
    if not command_exists("mlnx_qos"):
        print("warning: mlnx_qos not found; skipping ConnectX QoS mapping", file=sys.stderr)
        return
    prio_tc = ["0"] * 8
    prio_tc[priority] = "1"
    tsa = ["vendor"] * 8
    tsa[1] = "strict"
    commands = [
        ["mlnx_qos", "-i", iface, "--trust=dscp"],
        ["mlnx_qos", "-i", iface, "--dscp2prio", f"set,{dscp},{priority}"],
        ["mlnx_qos", "-i", iface, "-p", ",".join(prio_tc)],
        ["mlnx_qos", "-i", iface, "-s", ",".join(tsa)],
        ["mlnx_qos", "-i", iface, "-f", "0,0,0,0,0,0,0,0"],
    ]
    for cmd in commands:
        run(cmd, check=False, dry_run=dry_run, warn=True)


def restore_mlnx_qos(
    iface: str,
    state: dict[str, Any] | None,
    dscp: int,
    priority: int,
    *,
    dry_run: bool,
) -> None:
    if not command_exists("mlnx_qos"):
        return
    values = state.get("parsed", {}).get("mlnx_qos", {}) if state else {}
    commands = [["mlnx_qos", "-i", iface, "--dscp2prio", f"del,{dscp},{priority}"]]
    if values.get("pfc"):
        commands.append(["mlnx_qos", "-i", iface, "-f", values["pfc"]])
    if values.get("prio_tc"):
        commands.append(["mlnx_qos", "-i", iface, "-p", values["prio_tc"]])
    if values.get("tsa"):
        commands.append(["mlnx_qos", "-i", iface, "-s", values["tsa"]])
    if values.get("trust"):
        commands.append(["mlnx_qos", "-i", iface, f"--trust={values['trust']}"])
    for cmd in commands:
        run(cmd, check=False, dry_run=dry_run, warn=True)


def apply_tc(iface: str, port: int, priority: int, *, dry_run: bool, hw_offload: bool) -> None:
    if not command_exists("tc"):
        print("warning: tc not found; skipping qdisc/filter setup", file=sys.stderr)
        return
    tx_queues = get_tx_queue_count(iface)
    if tx_queues < 2:
        raise CommandError(f"{iface} has {tx_queues} TX queues; mqprio profile needs at least 2")

    bulk_queues = tx_queues - 1
    priority_offset = bulk_queues
    priority_map = ["0"] * 16
    priority_map[priority] = "1"
    root_qdisc = output(["tc", "qdisc", "show", "dev", iface]).splitlines()
    if root_qdisc and root_qdisc[0].startswith("qdisc mqprio "):
        run(
            ["tc", "qdisc", "del", "dev", iface, "root"],
            check=False,
            dry_run=dry_run,
            capture=True,
        )
    run(
        [
            "tc",
            "qdisc",
            "replace",
            "dev",
            iface,
            "root",
            "mqprio",
            "num_tc",
            "2",
            "map",
            *priority_map,
            "queues",
            f"{bulk_queues}@0",
            f"1@{priority_offset}",
            "hw",
            "1" if hw_offload else "0",
        ],
        dry_run=dry_run,
    )
    run(["tc", "qdisc", "replace", "dev", iface, "clsact"], dry_run=dry_run)
    for pref in ("100", "101"):
        run(
            ["tc", "filter", "del", "dev", iface, "egress", "pref", pref],
            check=False,
            dry_run=dry_run,
            capture=True,
        )
    run(
        [
            "tc",
            "filter",
            "add",
            "dev",
            iface,
            "egress",
            "protocol",
            "ip",
            "pref",
            "100",
            "flower",
            "ip_proto",
            "udp",
            "dst_port",
            str(port),
            "action",
            "skbedit",
            "priority",
            str(priority),
        ],
        dry_run=dry_run,
    )
    run(
        [
            "tc",
            "filter",
            "add",
            "dev",
            iface,
            "egress",
            "protocol",
            "ip",
            "pref",
            "101",
            "flower",
            "ip_proto",
            "udp",
            "src_port",
            str(port),
            "action",
            "skbedit",
            "priority",
            str(priority),
        ],
        dry_run=dry_run,
    )


def restore_tc(iface: str, *, dry_run: bool, remove_root: bool) -> None:
    if not command_exists("tc"):
        return
    for pref in ("100", "101"):
        run(
            ["tc", "filter", "del", "dev", iface, "egress", "pref", pref],
            check=False,
            dry_run=dry_run,
            capture=True,
        )
    if remove_root:
        run(["tc", "qdisc", "del", "dev", iface, "root"], check=False, dry_run=dry_run, capture=True)
    else:
        print("warning: no restore point; skipping root qdisc reset", file=sys.stderr)


def read_sysctls() -> dict[str, str]:
    values: dict[str, str] = {}
    if not command_exists("sysctl"):
        return values
    for key in SYSCTL_TUNABLES:
        result = run(["sysctl", "-n", key], check=False, capture=True)
        if result.returncode == 0:
            values[key] = (result.stdout or "").strip()
    return values


def apply_sysctls(*, dry_run: bool) -> None:
    if not command_exists("sysctl"):
        print("warning: sysctl not found; skipping kernel latency knobs", file=sys.stderr)
        return
    for key, value in SYSCTL_TUNABLES.items():
        run(["sysctl", "-w", f"{key}={value}"], check=False, dry_run=dry_run, warn=True)


def restore_sysctls(state: dict[str, Any] | None, *, dry_run: bool) -> None:
    if not command_exists("sysctl") or state is None:
        return
    values = state.get("parsed", {}).get("sysctl", {})
    for key, value in values.items():
        run(["sysctl", "-w", f"{key}={value}"], check=False, dry_run=dry_run, warn=True)


def install_linuxptp_if_needed(*, dry_run: bool, install: bool) -> bool:
    if command_exists("ptp4l") and command_exists("phc2sys"):
        return True
    if not install:
        print("warning: linuxptp tools not found; skipping PTP/IEEE1588", file=sys.stderr)
        return False
    if not command_exists("apt-get"):
        print("warning: apt-get not found; cannot install linuxptp", file=sys.stderr)
        return False
    run(["apt-get", "update"], dry_run=dry_run)
    run(["apt-get", "install", "-y", "linuxptp"], dry_run=dry_run)
    return dry_run or (command_exists("ptp4l") and command_exists("phc2sys"))


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def start_background(name: str, iface: str, cmd: list[str], *, dry_run: bool) -> None:
    path = pid_path(iface, name)
    if path.exists():
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if pid > 0 and process_running(pid):
            print(f"{name} already running for {iface}: pid {pid}")
            return
    if dry_run:
        print(f"+ start {name}: {quote_cmd(cmd)} > {log_path(iface, name)} 2>&1 &")
        print(f"+ write pid {path}")
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = log_path(iface, name).open("ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"started {name} pid {proc.pid}; log {log_path(iface, name)}")


def stop_background(name: str, iface: str, *, dry_run: bool) -> None:
    path = pid_path(iface, name)
    if not path.exists():
        return
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = -1
    if pid <= 0:
        if not dry_run:
            path.unlink(missing_ok=True)
        return
    if dry_run:
        print(f"+ stop {name} pid {pid}")
        print(f"+ remove pid {path}")
        return
    if process_running(pid):
        with contextlib.suppress(OSError):
            os.killpg(pid, 15)
    path.unlink(missing_ok=True)
    print(f"stopped {name} pid {pid}")


def apply_ptp(
    iface: str,
    *,
    dry_run: bool,
    install: bool,
    transport: str,
    domain: int,
    dscp: int,
    priority: int,
    slave_only: bool,
) -> None:
    if not install_linuxptp_if_needed(dry_run=dry_run, install=install):
        return
    transport_flag = "-2" if transport == "l2" else "-4"
    ptp4l_cmd = [
        "ptp4l",
        "-i",
        iface,
        "-m",
        "-H",
        transport_flag,
        "--domainNumber",
        str(domain),
        "--dscp_event",
        str(dscp),
        "--dscp_general",
        str(dscp),
        "--socket_priority",
        str(priority),
    ]
    if slave_only:
        ptp4l_cmd.extend(["--clientOnly", "1"])
    phc2sys_cmd = ["phc2sys", "-a", "-r", "-w", "-m", "-n", str(domain)]
    start_background("ptp4l", iface, ptp4l_cmd, dry_run=dry_run)
    start_background("phc2sys", iface, phc2sys_cmd, dry_run=dry_run)


def restore_ptp(iface: str, *, dry_run: bool) -> None:
    stop_background("phc2sys", iface, dry_run=dry_run)
    stop_background("ptp4l", iface, dry_run=dry_run)


def ptp_status_lines(iface: str) -> list[str]:
    lines: list[str] = []
    for name in ("ptp4l", "phc2sys"):
        path = pid_path(iface, name)
        if not path.exists():
            lines.append(f"{name}: stopped")
            continue
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        state = "running" if pid > 0 and process_running(pid) else "stale"
        lines.append(f"{name}: {state} pid={pid} log={log_path(iface, name)}")
    return lines


def ensure_iface(iface: str) -> None:
    if not iface_exists(iface):
        raise CommandError(f"interface does not exist: {iface}")


def apply_profile(args: argparse.Namespace) -> None:
    ensure_iface(args.iface)
    require_root_for_mutation(args)
    save_snapshot(args.iface, args.profile, replace=args.replace_state, dry_run=args.dry_run)
    if args.profile == "lab":
        apply_sysctls(dry_run=args.dry_run)
        apply_private_flags(args.iface, PTP_PRIVATE_FLAGS, dry_run=args.dry_run)
    apply_coalesce(args.iface, dry_run=args.dry_run)
    apply_pause(args.iface, dry_run=args.dry_run, keep_pause=args.keep_pause)
    apply_mlnx_qos(args.iface, args.dscp, args.priority, dry_run=args.dry_run)
    for direction in ("dst", "src"):
        add_iptables_rule(args.port, direction, args.dscp, dry_run=args.dry_run)
    apply_tc(
        args.iface,
        args.port,
        args.priority,
        dry_run=args.dry_run,
        hw_offload=args.hw_offload,
    )
    if args.ptp:
        apply_ptp(
            args.iface,
            dry_run=args.dry_run,
            install=args.install_ptp,
            transport=args.ptp_transport,
            domain=args.ptp_domain,
            dscp=args.dscp,
            priority=args.priority,
            slave_only=args.ptp_slave_only,
        )
    print(
        f"TSN-like probe profile enabled on {args.iface}: "
        f"UDP/{args.port}, DSCP {args.dscp}, priority {args.priority}, profile {args.profile}"
    )


def restore_profile(args: argparse.Namespace) -> None:
    ensure_iface(args.iface)
    require_root_for_mutation(args)
    state = load_snapshot(args.iface)
    if state is None:
        print(
            f"warning: no restore point found at {state_path(args.iface)}; "
            "removing owned rules and leaving unknown NIC settings unchanged",
            file=sys.stderr,
        )
    restore_ptp(args.iface, dry_run=args.dry_run)
    restore_tc(args.iface, dry_run=args.dry_run, remove_root=state is not None or args.force_root_reset)
    for direction in ("dst", "src"):
        delete_iptables_rule(args.port, direction, args.dscp, dry_run=args.dry_run)
    restore_mlnx_qos(args.iface, state, args.dscp, args.priority, dry_run=args.dry_run)
    restore_pause(args.iface, state, dry_run=args.dry_run)
    restore_coalesce(args.iface, state, dry_run=args.dry_run)
    restore_private_flags(args.iface, state, dry_run=args.dry_run)
    restore_sysctls(state, dry_run=args.dry_run)
    if state is not None and not args.keep_state:
        delete_snapshot(args.iface, dry_run=args.dry_run)
    print(f"TSN-like probe profile disabled on {args.iface}")


def print_status(args: argparse.Namespace) -> None:
    ensure_iface(args.iface)
    print(f"Interface: {args.iface}")
    print(f"State file: {state_path(args.iface)}")
    print(f"State exists: {'yes' if state_path(args.iface).exists() else 'no'}")
    print()
    sections = [
        ("qdisc", ["tc", "qdisc", "show", "dev", args.iface]),
        ("egress filters", ["tc", "filter", "show", "dev", args.iface, "egress"]),
        ("coalescing", ["ethtool", "-c", args.iface]),
        ("pause", ["ethtool", "-a", args.iface]),
        ("private flags", ["ethtool", "--show-priv-flags", args.iface]),
        ("mlx qos", ["mlnx_qos", "-i", args.iface]),
    ]
    for title, cmd in sections:
        if not command_exists(cmd[0]):
            continue
        print(f"== {title} ==")
        text = output(cmd).strip()
        print(text or "(none)")
        print()
    print("== ptp ==")
    print("\n".join(ptp_status_lines(args.iface)))
    print()
    if command_exists("iptables"):
        print("== owned iptables counters ==")
        counters = [
            line
            for line in output(["iptables", "-t", "mangle", "-vnL", "OUTPUT"]).splitlines()
            if COMMENT_PREFIX in line
        ]
        print("\n".join(counters) if counters else "(none)")
        print()
    if command_exists("iptables-save"):
        print("== owned iptables rules ==")
        rules = [
            line
            for line in output(["iptables-save", "-t", "mangle"]).splitlines()
            if COMMENT_PREFIX in line
        ]
        print("\n".join(rules) if rules else "(none)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn TerraMeta TSN-like UDP probe networking features on/off."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--iface",
            default=os.environ.get("TM_NET_IFACE", "enP2p1s0f1np1"),
            help="network interface to tune (default: TM_NET_IFACE or enP2p1s0f1np1)",
        )
        subparser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP probe port")
        subparser.add_argument("--dscp", type=int, default=DEFAULT_DSCP, help="DSCP value to mark")
        subparser.add_argument(
            "--priority",
            type=int,
            default=DEFAULT_PRIORITY,
            choices=range(0, 8),
            metavar="0-7",
            help="Linux/NIC priority to assign",
        )

    apply_parser = subparsers.add_parser("apply", help="enable the TSN-like probe profile")
    add_common(apply_parser)
    apply_parser.add_argument(
        "--profile",
        default=os.environ.get("TM_TSN_PROFILE", "lab"),
        choices=("priority", "lab"),
        help="priority = network QoS only; lab = QoS plus aggressive host/PTP tuning",
    )
    apply_parser.add_argument("--dry-run", action="store_true", help="print commands without changing state")
    apply_parser.add_argument("--replace-state", action="store_true", help="overwrite existing restore point")
    apply_parser.add_argument("--keep-pause", action="store_true", help="do not change Ethernet pause settings")
    apply_parser.add_argument("--hw-offload", action="store_true", help="request mqprio hardware offload")
    apply_parser.add_argument(
        "--no-ptp",
        dest="ptp",
        action="store_false",
        default=True,
        help="do not start ptp4l/phc2sys",
    )
    apply_parser.add_argument(
        "--no-install-ptp",
        dest="install_ptp",
        action="store_false",
        default=True,
        help="do not install linuxptp automatically when missing",
    )
    apply_parser.add_argument(
        "--ptp-transport",
        default=os.environ.get("TM_PTP_TRANSPORT", "l2"),
        choices=("l2", "udp4"),
        help="PTP transport for ptp4l",
    )
    apply_parser.add_argument(
        "--ptp-domain",
        type=int,
        default=int(os.environ.get("TM_PTP_DOMAIN", "0")),
        help="PTP domain number",
    )
    apply_parser.add_argument(
        "--ptp-slave-only",
        action="store_true",
        help="start ptp4l in slave-only mode; do not use this on both ends of a direct link",
    )
    apply_parser.set_defaults(func=apply_profile)

    restore_parser = subparsers.add_parser("restore", help="disable the profile and restore saved settings")
    add_common(restore_parser)
    restore_parser.add_argument("--dry-run", action="store_true", help="print commands without changing state")
    restore_parser.add_argument("--keep-state", action="store_true", help="keep the restore-point JSON")
    restore_parser.add_argument(
        "--force-root-reset",
        action="store_true",
        help="reset the root qdisc even when no restore point exists",
    )
    restore_parser.set_defaults(func=restore_profile)

    status_parser = subparsers.add_parser("status", help="show current qdisc/QoS/profile state")
    status_parser.add_argument(
        "--iface",
        default=os.environ.get("TM_NET_IFACE", "enP2p1s0f1np1"),
        help="network interface to inspect (default: TM_NET_IFACE or enP2p1s0f1np1)",
    )
    status_parser.set_defaults(func=print_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
