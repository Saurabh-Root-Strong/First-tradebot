# Migrate Tradebot VM: AWS t3.micro → Oracle Cloud A1 (free, 24 GB RAM)

Why: the t3.micro's 1 GB RAM OOM-kills the dashboard in a loop, which starves the
parquet mirrors (esp. `oi_snapshots`) so every OI-gated panel stays "warming up".
Oracle's **Always Free Ampere A1** gives **4 vCPU + 24 GB RAM for $0** → OOM gone,
mirrors fill, panels populate. This replaces the t3.small plan
([aws_t3small_runbook.md](aws_t3small_runbook.md) — now moot if we go Oracle).

Arch note: A1 is **ARM64 (aarch64)**. The Docker image rebuilds from source/wheels
for arm64 on the new box — pandas/numpy/pyarrow/numba all ship aarch64 wheels, so a
plain `docker compose build` works. The **only real risk is the build** (see Step 6).

---

## Step 1 — Oracle account + capacity

1. Sign up at cloud.oracle.com (Free Tier; card for identity check, not charged).
   Pick home region **India South (Hyderabad)** or **India West (Mumbai)**.
2. **Upgrade to Pay-As-You-Go** (Billing → Upgrade). Always-Free resources stay
   free; PAYG dramatically improves A1 capacity success (pure-free accounts hit
   "Out of capacity" constantly).

## Step 2 — Create the A1 instance

Compute → Instances → **Create instance**:
- Image: **Ubuntu 24.04 (aarch64/ARM)**.
- Shape: **Ampere → VM.Standard.A1.Flex**, **4 OCPUs, 24 GB** (all within Always-Free).
- Networking: create/allow a **VCN with a public subnet** (default wizard is fine);
  **Assign public IP = yes**.
- SSH keys: paste your existing public key
  (`ssh-keygen -y -f ~/Downloads/tradebot-key.pem` to print it) OR add a new one.
- If you hit **"Out of host capacity"**: retry every few min, try another
  Availability Domain, or run a create loop via the OCI CLI. PAYG + off-peak hours help.

> Optional stable IP: Networking → **Reserved public IPs** → reserve one → attach to
> the instance's VNIC. Free while attached. Avoids IP churn on stop/start.

## Step 3 — Open the firewall (TWO layers — Oracle gotcha)

Oracle blocks ports at BOTH the cloud SG and the OS iptables.

1. **Security List** (VCN → Subnet → Security List) → add Ingress rules:
   `0.0.0.0/0` TCP **22, 80, 443**.
2. **OS iptables** (Oracle Ubuntu ships a default REJECT) — SSH in and:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

## Step 4 — Base setup on the A1 box

```bash
ssh -i ~/Downloads/tradebot-key.pem ubuntu@<A1_IP>
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu && newgrp docker      # re-login after
free -m                                              # confirm ~24000 total
```

## Step 5 — Bring the app over

```bash
git clone <your repo URL> ~/tradebot           # or scp the repo up
cd ~/tradebot
git checkout feature/shock-layer-trade-mgmt
```
Copy the secrets/infra the repo doesn't track (from the AWS box or laptop), scp to
`~/tradebot/` on A1:
- `access_token.txt`
- `.env` (the `BASIC_AUTH_HASH`)
- `Caddyfile`, `docker-compose.yml` (VM-local versions on the AWS box)
- existing `data/` if you want history seeded (optional — eod_sync also has it locally)

## Step 6 — Build for ARM64 + run

```bash
cd ~/tradebot
docker compose build          # rebuilds wheels for aarch64 — this is the risk step
docker compose up -d
docker compose logs -f tradebot
```
If the build fails on a package (most likely **numba/llvmlite**):
- ensure the Dockerfile base Python is a version with arm64 wheels (3.11/3.12),
- or `sudo apt-get install -y build-essential llvm` and rebuild,
- numpy already pinned `==2.2.6` for numba (keep it).

Confirm: `free -m` shows tons of headroom, no `code -9` in logs, and after ~20 min
the `oi_snapshots` mirror grows (no longer 7 KB).

## Step 7 — Caddy / hostname

Point Caddy at `<A1_IP>.sslip.io` (same pattern as now) or your reserved-IP/duckdns
name. `tls internal` only if using a bare IP; with sslip.io you get a real LE cert.

## Step 8 — Rewire the client side (send me the A1 IP, I do this)

Same 4 repo files + token push as the AWS plan:
- `sync_from_vm.py` `DEF_HOST` (drives `eod_sync.py`, `view_vm.bat`)
- `morning_token.bat` `VM=` + open-URL
- `push_news.bat`
- `DEPLOY.md`
- VM `~/tradebot/Caddyfile` hostname

## Step 9 — Decommission AWS

Once A1 is verified capturing a full clean session: **stop** the t3.micro (keep it a
few days as fallback), then terminate + release its EBS/EIP to stop billing.

---

### Net result
24 GB RAM, $0/mo, no OOM, no scheduling/EIP juggling, mirrors fill, all panels
populate. The morning-token + eod-sync laptop tasks keep working unchanged (just the
host IP swaps).
