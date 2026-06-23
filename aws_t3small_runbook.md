# t3.small upgrade + scheduled stop/start (cheaper-than-or-equal, OOM-proof)

Goal: move the VM to **t3.small (2 GB)** so the dashboard stops OOM-ing, and
auto-stop it nights + weekends so the bill stays ~equal to the micro-24×7 cost.

**Do the whole thing TONIGHT after 15:30 IST** (and after the day's `eod_sync`
has archived today) — the resize needs a Stop, which would interrupt live capture
during market hours.

Region: **ap-south-1 (Mumbai)**. Instance: the one at the current IP
`13.233.88.148` (find its **Instance ID** `i-xxxxxxxx` in EC2 → Instances).

---

## Step 1 — Elastic IP (stable address across stop/start)

A stopped/started instance gets a NEW random public IP unless an Elastic IP (EIP)
is attached. Do this FIRST so the address settles before everything else.

1. EC2 → **Network & Security → Elastic IPs → Allocate Elastic IP address** → Allocate.
2. Select it → **Actions → Associate** → Resource type **Instance** → pick the
   instance → Associate.
3. **Note the new EIP** (e.g. `3.x.x.x`). The moment it associates, the old
   `13.233.88.148` is released — from now the VM lives at the EIP.

> Cost note: EIP is free while attached to a *running* instance; ~$0.005/h while
> the instance is *stopped*. That's the ~$2/mo in the estimate. (To avoid it
> entirely, use duckdns instead — see Step 5, optional.)

After you have the EIP, send it to me — I'll swap it into the 4 repo files
(`sync_from_vm.py`, `morning_token.bat`, `push_news.bat`, `DEPLOY.md`) and the VM
Caddyfile in one go.

---

## Step 2 — Resize micro → small

1. EC2 → Instances → select it → **Instance state → Stop instance** (wait: Stopped).
2. **Actions → Instance settings → Change instance type** → **t3.small** → Apply.
3. **Instance state → Start instance**.
4. SSH in, confirm RAM doubled and bring the stack up:
   ```bash
   ssh -i ~/Downloads/tradebot-key.pem ubuntu@<EIP>
   free -m            # total should be ~1960, not ~908
   sudo sysctl vm.swappiness=10      # keep the safe value
   cd ~/tradebot && docker compose up -d
   ```

---

## Step 3 — IAM role for the scheduler

EventBridge Scheduler needs permission to start/stop the instance.

1. IAM → **Roles → Create role** → Trusted entity **Custom trust policy**, paste:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": { "Service": "scheduler.amazonaws.com" },
       "Action": "sts:AssumeRole"
     }]
   }
   ```
2. Attach a permission policy (Create policy → JSON):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["ec2:StartInstances", "ec2:StopInstances"],
       "Resource": "arn:aws:ec2:ap-south-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>"
     }]
   }
   ```
3. Name the role e.g. `tradebot-scheduler-role`. Note its ARN.

---

## Step 4 — Two EventBridge schedules (start + stop)

EventBridge → **Scheduler → Schedules → Create schedule**. Make TWO.

Both: Schedule type **Recurring**, **Cron-based**, Timezone **Asia/Kolkata**,
Flexible time window **Off**. Target = **Templated → EC2 → StartInstances /
StopInstances**, Instance ID = `<INSTANCE_ID>`, Execution role = the role from Step 3.

| Schedule name      | Cron                         | Action         |
|--------------------|------------------------------|----------------|
| `tradebot-start`   | `cron(30 8 ? * MON-FRI *)`   | StartInstances |
| `tradebot-stop`    | `cron(30 23 ? * MON-FRI *)`  | StopInstances  |

- Start **08:30** = buffer before the 09:00 token task + 09:15 open.
- Stop **23:30** = leaves the whole evening for your laptop's `eod_sync` to pull
  + archive while the VM is still up. (If you reliably sync earlier, move the stop
  earlier to save more.)
- Weekends: no schedule fires → stays stopped Sat/Sun.

> The morning token task (09:00) already restarts the container; with the VM
> started at 08:30 it'll be up in time. If a morning the VM is somehow still
> stopped, just start it from the console (or run morning_token after a manual start).

---

## Step 5 — (optional) duckdns instead of EIP, to drop the ~$2/mo

Skip Step 1's EIP; instead give the VM a free stable hostname that it self-updates
on every boot:

1. duckdns.org → sign in → create a subdomain, copy your token.
2. On the VM, a boot hook pushes the current IP:
   ```bash
   # /etc/cron.d/duckdns  (or a @reboot crontab line)
   @reboot root curl -s "https://www.duckdns.org/update?domains=<SUB>&token=<TOKEN>&ip=" >/dev/null
   ```
3. Point Caddy at `<SUB>.duckdns.org` (replaces the sslip hostname) so Let's
   Encrypt certs that name. Then rewire the repo files to `<SUB>.duckdns.org`
   instead of an IP.

Trade-off: no EIP charge + still stable, but the cert hostname/Caddyfile change is
extra one-time work.

---

## After: what I update in code (send me the EIP or duckdns name)

- `sync_from_vm.py` `DEF_HOST` (also used by `eod_sync.py`, `view_vm.bat`)
- `morning_token.bat` `VM=` and the open-URL
- `push_news.bat`
- `DEPLOY.md`
- VM `~/tradebot/Caddyfile` hostname
- (these are the only 4 repo files + 1 VM file that hardcode the address)
