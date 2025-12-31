# Getting Started with Athena SQL Optimizer (Beginner's Guide)

> **What is this?** This tool helps you find slow and expensive SQL queries in AWS Athena, and tells you exactly how to fix them to save money. It works like a spell-checker for database queries!

**Time needed:** 30-45 minutes
**Skill level:** Beginner-friendly (no coding required)
**What you'll save:** Potentially 50-80% on your AWS Athena bills!

---

## 📋 What You'll Need (Prerequisites)

Before starting, make sure you have these. If not, follow the links to set them up:

### 1. **AWS Account** (Required)
- **What it is:** Amazon's cloud service where your data lives
- **Do you have it?** Can you log into https://console.aws.amazon.com?
- **Don't have it?** Create one at https://aws.amazon.com/free/
  - Takes 10 minutes
  - Requires credit card (but won't charge for small usage)

### 2. **Claude Desktop** (Required)
- **What it is:** The application that will use this optimizer
- **Do you have it?** Look for "Claude" app on your computer
- **Don't have it?** Download from https://claude.ai/download
  - Available for Mac, Windows, and Linux
  - Free to use

### 3. **Python 3.10 or newer** (Required)
- **What it is:** A programming language (you won't write code, it just needs to be installed)
- **Check if you have it:**
  - Open Terminal (Mac) or Command Prompt (Windows)
  - Type: `python --version`
  - If you see "Python 3.10" or higher, you're good!
- **Don't have it?** Download from https://www.python.org/downloads/
  - Click the big yellow "Download" button
  - Run the installer (check "Add Python to PATH" on Windows!)

### 4. **Git** (Required)
- **What it is:** A tool to download code from the internet
- **Check if you have it:**
  - Open Terminal/Command Prompt
  - Type: `git --version`
  - If you see "git version...", you're good!
- **Don't have it?** Download from https://git-scm.com/downloads
  - Choose your operating system
  - Use all default settings during install

### 5. **AWS CLI** (Required)
- **What it is:** A tool to connect to your AWS account
- **Check if you have it:**
  - Type: `aws --version`
- **Don't have it?** Follow: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
  - Takes 5 minutes

---

## 🔧 Step 1: Set Up Your AWS Account (10 minutes)

### 1.1 Create an S3 Bucket for Query Results

AWS Athena needs a place to store results. Think of this like a folder in the cloud.

**Steps:**
1. Log into AWS Console: https://console.aws.amazon.com
2. In the search bar at the top, type **"S3"** and click on it
3. Click the orange **"Create bucket"** button
4. Give it a name like: `my-athena-results-2024` (must be unique worldwide)
5. Choose your **region** (e.g., "US East (N. Virginia)" or "EU (Ireland)")
   - **Write this down!** You'll need it later
6. Leave everything else as default
7. Click **"Create bucket"** at the bottom

**What you'll need later:**
- Bucket name: `my-athena-results-2024`
- Region: `us-east-1` (or whatever you chose)

### 1.2 Find Your Athena Workgroup

A workgroup is like a folder that organizes your queries.

**Steps:**
1. In AWS Console, search for **"Athena"** and click it
2. Click **"Workgroups"** in the left menu
3. You'll see a list - usually there's one called **"primary"**
4. **Write this down!** You'll need it for configuration

### 1.3 Set Up AWS Credentials

This lets the tool access your AWS account.

**Steps:**
1. In AWS Console, search for **"IAM"** (Identity and Access Management)
2. Click **"Users"** in the left menu
3. Click your username (or create a new user if needed)
4. Click **"Security credentials"** tab
5. Scroll to **"Access keys"** section
6. Click **"Create access key"**
7. Choose **"Command Line Interface (CLI)"**
8. Check the box confirming you understand
9. Click **"Next"** then **"Create access key"**
10. **IMPORTANT:** Copy both values:
    - Access key ID: `AKIAIOSFODNN7EXAMPLE`
    - Secret access key: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
11. Keep these safe and secret!

**Configure AWS CLI:**
1. Open Terminal/Command Prompt
2. Type: `aws configure`
3. Paste your Access Key ID when asked
4. Paste your Secret Access Key when asked
5. Enter your region (e.g., `us-east-1`)
6. Press Enter for output format (default is fine)

**Test it works:**
```bash
aws s3 ls
```
You should see your buckets listed!

---

## 💻 Step 2: Install the Optimizer (5 minutes)

### 2.1 Download the Code

**Steps:**
1. Open Terminal (Mac) or Command Prompt (Windows)
2. Go to your Documents folder:
   ```bash
   cd ~/Documents
   ```
3. Download the code:
   ```bash
   git clone https://github.com/YOUR-USERNAME/athena-sql-optimizer.git
   ```
4. Go into the folder:
   ```bash
   cd athena-sql-optimizer
   ```

### 2.2 Install Dependencies

This installs all the helper tools needed.

**Steps:**
1. Install uv (a fast installer):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Install the optimizer:
   ```bash
   uv sync
   ```
3. Wait 1-2 minutes while it downloads everything

**You'll see:** Lots of text scrolling. That's normal! Wait for it to finish.

---

## ⚙️ Step 3: Configure Claude Desktop (10 minutes)

### 3.1 Find Your Config File

Claude Desktop needs to know about the optimizer.

**On Mac:**
```bash
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**On Windows:**
```bash
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**On Linux:**
```bash
nano ~/.config/Claude/claude_desktop_config.json
```

### 3.2 Add the Optimizer Configuration

Copy this ENTIRE block and paste it in the file (replace the example values with YOUR values):

```json
{
  "mcpServers": {
    "athena-optimizer": {
      "command": "python",
      "args": [
        "-m",
        "athena_optimizer.server"
      ],
      "env": {
        "AWS_PROFILE": "default",
        "AWS_REGION": "us-east-1",
        "ATHENA_WORKGROUP": "primary",
        "ATHENA_S3_OUTPUT": "s3://my-athena-results-2024/",
        "ATHENA_DATABASE": "my_database",
        "RUN_EXPLAIN_ANALYZE": "false",
        "ATHENA_COST_PER_TB": "5.0",
        "TIMEOUT_SECONDS": "300"
      }
    }
  }
}
```

**What to change:**
- `AWS_REGION`: Your region from Step 1.1 (e.g., `us-east-1`)
- `ATHENA_S3_OUTPUT`: Your bucket name from Step 1.1 (must start with `s3://` and end with `/`)
- `ATHENA_DATABASE`: Your database name (find this in Athena console)

**Save the file:**
- Mac: Press `Cmd+S`
- Windows: Press `Ctrl+S`
- Linux: Press `Ctrl+X`, then `Y`, then Enter

### 3.3 Restart Claude Desktop

1. Quit Claude Desktop completely (not just close the window)
2. Open it again
3. Look for a new hammer icon (🔨) in the bottom right
4. Click it - you should see "athena-optimizer" listed!

---

## 🚀 Step 4: Your First Query Analysis (5 minutes)

Let's test it with a real example!

### 4.1 Analyze a Query

In Claude Desktop, type this:

```
Please analyze this SQL query for me:

SELECT *
FROM my_database.sales_data
WHERE year = 2024
```

### 4.2 What You'll See

Claude will use the optimizer tool and show you something like:

```
🔍 Analysis Results:

❌ CRITICAL Issue: Using SELECT *
   - Current cost: $5.50
   - Optimized cost: $1.10
   - Savings: $4.40 (80%)

   Fix: Only select the columns you need:
   SELECT customer_id, amount, date
   FROM my_database.sales_data
   WHERE year = 2024

⚠️ HIGH Issue: Missing partition filter
   - Add partition keys to your WHERE clause for better performance

📊 Total Savings Possible: $4.40 per query (80% reduction)
```

### 4.3 Understanding the Output

**Severity Levels:**
- 🔴 **CRITICAL**: Fix immediately (high cost impact)
- 🟠 **HIGH**: Fix soon (significant savings)
- 🟡 **MEDIUM**: Consider fixing (moderate savings)
- 🟢 **LOW**: Nice to have (small savings)
- ℹ️ **INFO**: Informational (no cost impact)

**For Each Issue, You'll See:**
- **What's wrong**: Plain English explanation
- **Current cost**: What you're paying now
- **Optimized cost**: What you'd pay after fixing
- **Savings**: How much money you'll save
- **How to fix**: Exact SQL code to use

---

## ✅ Step 5: Using the Other Tools

### 5.1 Estimate Query Cost (Without Running It)

```
Estimate the cost of this query:
SELECT * FROM large_table
```

This checks how expensive a query would be WITHOUT actually running it. Great for testing!

### 5.2 Check Table Health

```
Check the health of my table: database_name.table_name
```

This tells you if your table is set up efficiently (format, partitions, compression).

### 5.3 Get Server Diagnostics

```
Show me the optimizer diagnostics
```

This shows cache performance and health status. Useful if things seem slow.

---

## 🆘 Troubleshooting (Common Issues)

### "Optimizer not initialized"
**Cause:** Configuration is wrong
**Fix:**
1. Check your `claude_desktop_config.json` file
2. Make sure `ATHENA_WORKGROUP` and `ATHENA_S3_OUTPUT` are set
3. Restart Claude Desktop

### "Access Denied" or "Permission denied"
**Cause:** AWS credentials not configured
**Fix:**
1. Run `aws configure` again
2. Make sure you entered the keys correctly
3. Check AWS IAM permissions (see Step 1.3)

### "Table not found"
**Cause:** Database or table name is wrong
**Fix:**
1. Log into AWS Athena console
2. Check the exact database and table names
3. Database names are case-sensitive!

### "Module not found" error
**Cause:** Installation didn't complete
**Fix:**
1. Go back to the optimizer folder:
   ```bash
   cd ~/Documents/athena-sql-optimizer
   ```
2. Run installation again:
   ```bash
   uv sync
   ```

### Tool doesn't appear in Claude
**Cause:** Config file wrong or Claude not restarted
**Fix:**
1. Check config file syntax (must be valid JSON)
2. Quit Claude completely (check Activity Monitor/Task Manager)
3. Start Claude again

---

## 📚 Next Steps

Now that you're set up, try these:

1. **Analyze your most expensive queries**
   - Go to AWS Athena console
   - Look at "Recent queries"
   - Copy the expensive ones and analyze them

2. **Check all your tables**
   - Use `check_table_health` on each table
   - Follow the recommendations to save money

3. **Set up regular monitoring**
   - Analyze queries weekly
   - Track your cost savings

4. **Learn more**
   - Read the full README.md for advanced features
   - Join AWS forums to learn about Athena best practices

---

## 💡 Quick Reference Card

**Analyze a query:**
```
Analyze this query: SELECT * FROM table WHERE...
```

**Estimate cost:**
```
Estimate cost: SELECT * FROM table
```

**Check table:**
```
Check table health: database.table
```

**Get diagnostics:**
```
Show diagnostics
```

---

## 🎉 Success!

You're now ready to optimize your AWS Athena queries and save money!

**Questions?**
- Check the troubleshooting section above
- Review the full README.md for advanced topics
- Ask Claude for help with specific queries

**Saving money?**
- Track your costs in AWS Cost Explorer
- Compare before/after implementing recommendations
- Most users save 50-80% on Athena costs!

---

*Last updated: December 2024*
