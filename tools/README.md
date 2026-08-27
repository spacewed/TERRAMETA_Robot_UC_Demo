# TSN/PTP Probe Network Tools

These scripts turn a low-jitter network profile on and off for the TerraMeta UDP latency probe. The default `lab` profile is intentionally probe-first: it prioritizes the small UDP probe packets on port `10000`, applies aggressive host latency knobs, enables high-quality PTP transmit timestamps when the NIC supports them, and starts PTP/IEEE 1588 when possible so you can measure the best achievable network timing before changing the high-bandwidth camera frame streams.

This is still not a full `taprio` time-aware gate schedule. It combines TSN-adjacent QoS, ConnectX traffic-class mapping, host scheduling/socket tuning, and PTP clock synchronization. It does not change the TCP camera streams.

## Quick Start

Run the same commands on both DGX systems, using the active ConnectX interface name on each host:

```bash
# Preview all commands without changing the NIC.
./tools/tsn_on.sh enP2p1s0f1np1 --dry-run

# Enable the maximum low-jitter lab profile.
sudo ./tools/tsn_on.sh enP2p1s0f1np1

# Inspect qdisc, QoS, coalescing, pause, private flags, PTP, and owned packet-marking rules.
./tools/tsn_status.sh enP2p1s0f1np1

# Run the normal receiver/sender visualizer test.
python3 On_Receiver/receiver.py
python3 On_Robot/sender.py

# Stop PTP and restore the NIC/kernel settings saved when the profile was enabled.
sudo ./tools/tsn_off.sh enP2p1s0f1np1
```

You can also set the interface once:

```bash
export TM_NET_IFACE=enP2p1s0f1np1
sudo ./tools/tsn_on.sh
./tools/tsn_status.sh
sudo ./tools/tsn_off.sh
```

## What Gets Changed

`tsn_on.sh` calls `tsn_probe_profile.py apply --profile lab`. It saves a restore point under `/var/tmp/terrameta_tsn/<iface>.json`, then applies these settings:

| Area | Change | Why |
| --- | --- | --- |
| Packet marking | Adds `iptables` mangle rules for UDP source/destination port `10000`, DSCP `46` | Marks the probe as expedited forwarding traffic |
| Linux priority | Adds `tc` egress filters that set `skb->priority` to `6` for UDP port `10000` | Lets qdisc/NIC queue selection see the probe as high priority |
| Transmit queues | Replaces the root qdisc with `mqprio`, mapping priority `6` to a dedicated transmit queue | Keeps probe packets out of the bulk traffic queue |
| ConnectX QoS | Uses `mlnx_qos` to trust DSCP, map DSCP `46` to priority `6`, and map priority `6` to a strict traffic class | Helps the NIC and switch-facing priority model agree |
| Interrupt moderation | Uses `ethtool -C` to disable adaptive moderation and set RX/TX usecs to `0`, frames to `1` | Reduces batching delay and jitter |
| Ethernet pause | Uses `ethtool -A` to turn global RX/TX pause off by default | Avoids pause-frame stalls during latency tests |
| Host latency sysctls | Sets `net.core.busy_read=50`, `net.core.busy_poll=50`, `kernel.timer_migration=0`, `kernel.sched_rt_runtime_us=-1` | Allows busy polling and avoids scheduler/timer behavior that can add probe spikes |
| PTP TX timestamping | Enables the ConnectX `tx_port_ts` private flag when supported | Improves PTP TX timestamp quality under load |
| PTP/IEEE 1588 | Installs `linuxptp` if missing, starts `ptp4l` on the interface, and starts `phc2sys -a -r` | Synchronizes clocks so the v2 probe can report PTP-backed one-way latency columns |

The profile is scoped to the latency probe. The camera frame TCP streams remain normal traffic.

The sender and receiver apps now also set UDP probe socket DSCP, `SO_PRIORITY`, larger probe socket buffers, optional busy-poll options, and v2 probe packets with `CLOCK_TAI` timestamps. One-way latency columns are recorded only when the `CLOCK_TAI` estimate passes an RTT sanity check; if PTP is not synchronized, those columns should stay empty instead of reporting misleading offset-inflated values. The apps attempt SCHED_FIFO priority and `mlockall`; those best-effort calls require root or equivalent capabilities, so they will warn and continue when run as an unprivileged user.

For camera-stream stability, keep the NIC/PTP profile enabled with `sudo ./tools/tsn_on.sh ...`, but start the robot sender as the normal venv user first. If the aggressive app-side probe settings interfere with camera capture or echo reception, use this integrated-probe sender command:

```bash
TM_NET_IFACE=enP2p1s0f1np1 \
TM_CAMERA_STARTUP_HARD_RESET=0 \
TM_CAMERA_SHUTDOWN_HARD_RESET=0 \
TM_PROBE_BIND_DEVICE=off \
TM_PROBE_CPU_CORE=off \
TM_PROBE_RT_PRIORITY=0 \
TM_PROBE_MLOCK=0 \
TM_PROBE_BUSY_POLL_US=0 \
./On_Robot/venv/bin/python On_Robot/sender.py
```

## What Gets Restored

`tsn_off.sh` calls `tsn_probe_profile.py restore`. If a restore point exists, it stops `phc2sys` and `ptp4l`, removes the owned `tc` filters, removes the `mqprio` root qdisc, deletes the owned `iptables` rules, restores `mlnx_qos` values, restores pause settings, restores coalescing settings, restores managed private flags, restores host sysctls captured in the state file, and removes the state file.

If no restore point exists, `tsn_off.sh` still removes the owned filters and packet-marking rules, but it does not reset an arbitrary root qdisc unless you pass:

```bash
sudo ./tools/tsn_off.sh enP2p1s0f1np1 --force-root-reset
```

## Useful Options

```bash
# Keep an existing restore point instead of overwriting it.
sudo ./tools/tsn_on.sh enP2p1s0f1np1

# Replace the saved restore point with the current NIC state.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --replace-state

# Leave Ethernet pause settings untouched.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --keep-pause

# Request mqprio hardware offload. The default uses software mqprio because it is more portable.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --hw-offload

# Skip PTP and only apply QoS/host latency tuning.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --no-ptp

# Do not auto-install linuxptp if ptp4l/phc2sys are missing.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --no-install-ptp

# Use UDP/IPv4 PTP transport instead of layer-2 PTP.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --ptp-transport udp4

# Force slave-only PTP. Do not set this on both endpoints of a direct link.
sudo ./tools/tsn_on.sh enP2p1s0f1np1 --ptp-slave-only

# Keep the restore JSON after turning the profile off.
sudo ./tools/tsn_off.sh enP2p1s0f1np1 --keep-state
```

The Python entry point exposes the same controls:

```bash
python3 tools/tsn_probe_profile.py apply --iface enP2p1s0f1np1 --dry-run
sudo python3 tools/tsn_probe_profile.py apply --iface enP2p1s0f1np1
python3 tools/tsn_probe_profile.py status --iface enP2p1s0f1np1
sudo python3 tools/tsn_probe_profile.py restore --iface enP2p1s0f1np1
```

## Switch Notes

For direct DGX-to-DGX testing, run `tsn_on.sh` on both hosts with the same PTP transport and domain. Leave both in default PTP mode so BMCA can elect one side as the grandmaster.

For switched testing, the switch must preserve and honor DSCP `46` or map it to the equivalent priority queue. Configure the switch for PTP/IEEE 1588 boundary-clock or transparent-clock behavior. Without switch QoS, the packets may still be marked by the hosts, but the fabric may treat them like normal traffic. Without switch PTP support, RTT/2 metrics still work, but one-way latency columns may remain empty or invalid.

PTP logs are written to:

```text
/var/tmp/terrameta_tsn/<iface>_ptp4l.log
/var/tmp/terrameta_tsn/<iface>_phc2sys.log
```

## Requirements

The profile uses standard Linux networking tools plus NVIDIA/Mellanox QoS tooling when present:

| Tool | Purpose |
| --- | --- |
| `tc` | `mqprio` qdisc and egress filters |
| `ethtool` | coalescing, pause, and PTP private-flag settings |
| `iptables` | DSCP marking for UDP probe packets |
| `mlnx_qos` | ConnectX DSCP, priority, traffic-class, and PFC settings |
| `ptp4l` | PTP/IEEE 1588 clock synchronization on the NIC |
| `phc2sys` | Synchronizes system time with the PTP clock selected by `ptp4l` |

Apply and restore need root privileges because they change NIC and kernel networking state and start/stop PTP daemons. Status does not require root, though non-root status may not be able to display all `iptables` details.

## Safety Notes

- Run `tsn_on.sh` before the normal visualizer test and `tsn_off.sh` afterward.
- Run `tsn_status.sh` after enabling to confirm the qdisc, filters, QoS state, and restore file.
- Use `--dry-run` before first use on a new system.
- The first `tsn_on.sh` call preserves the original restore point. Use `--replace-state` only when you intentionally want the current NIC state to become the new restore target.
- If your switch or fabric relies on global Ethernet pause, enable with `--keep-pause`.
- For the largest app-side scheduling effect, run sender and receiver with root or suitable capabilities so SCHED_FIFO, busy-poll, and `mlockall` can succeed.
