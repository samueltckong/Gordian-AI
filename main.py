import streamlit as st
import json
import requests
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

# Try to import Groq with error handling
try:
    from groq import Groq
except ImportError:
    st.error("Groq package not found. Please install: `pip install groq`")
    Groq = None

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for API keys, endpoints, and thresholds"""
    
    # Attempt to get API keys from Streamlit secrets first, then environment variables
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        BLACKBOX_API_KEY = st.secrets["BLACKBOX_API_KEY"]
    except (AttributeError, KeyError):
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        BLACKBOX_API_KEY = os.environ.get("BLACKBOX_API_KEY")

    # Ensure API keys are set, raise error if missing
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in Streamlit secrets or environment variables")
    if not BLACKBOX_API_KEY:
        raise ValueError("BLACKBOX_API_KEY is not set in Streamlit secrets or environment variables")

    STALE_DAYS_THRESHOLD = 365
    VERY_STALE_DAYS_THRESHOLD = 730
    NPM_REGISTRY_BASE = "https://registry.npmjs.org"
    NPM_API_BASE = "https://api.npmjs.org"
    GROQ_FAST_MODEL = "llama3-8b-8192"
    GROQ_DEEP_MODEL = "llama3-70b-8192"

# --- DATA MODELS ---
@dataclass
class PackageInfo:
    name: str
    version: str
    last_publish: Optional[datetime]
    deprecated: bool
    deprecated_message: Optional[str]
    description: Optional[str]
    dependencies_count: int
    weekly_downloads: int

@dataclass
class CodeUsage:
    file_path: str
    line_number: int
    code_snippet: str

@dataclass
class MigrationGuide:
    from_package: str
    to_package: str
    migration_steps: List[str]
    code_examples: Dict[str, str]
    complexity: str
    estimated_time: str

# --- BLACKBOX AI AGENT ---
class BlackboxAIAgent:
    """AI Agent for code generation and migration using BlackboxAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.blackbox.ai/api/chat"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
    def generate_migration_code(self, old_package: str, new_package: str, usage_examples: List[str] = None) -> Dict[str, str]:
        """Generate migration code examples using BlackboxAI"""
        
        # Build context from usage examples
        context = ""
        if usage_examples:
            context = "Current usage examples in the codebase:\n"
            for example in usage_examples[:5]:  # Limit to 5 examples
                context += f"```javascript\n{example}\n```\n"
        
        prompt = f"""You are an expert JavaScript developer specializing in npm package migrations.
        
Task: Generate complete, production-ready migration code for replacing '{old_package}' with '{new_package}'.

{context}

Generate the following:

1. Installation commands
2. Import statement changes
3. API migration mappings (show before/after for common patterns)
4. Complete working examples
5. Common pitfalls and solutions

Format your response as valid JSON with these keys:
- installation: object with remove and add commands
- imports: object with old and new import patterns
- api_mappings: array of {{old: "...", new: "...", note: "..."}}
- examples: array of {{title: "...", old_code: "...", new_code: "..."}}
- pitfalls: array of {{issue: "...", solution: "..."}}

Provide real, working code that developers can copy and use immediately."""

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "model": "blackboxai"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json().get('message', response.text)
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        # If no JSON found, structure the response
                        return self._structure_migration_response(content, old_package, new_package)
                except json.JSONDecodeError:
                    return self._structure_migration_response(content, old_package, new_package)
            else:
                logger.error(f"BlackboxAI API error: {response.status_code} - {response.text}")
                return {"error": f"API request failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"BlackboxAI generation error: {e}")
            return {"error": str(e)}
    
    def _structure_migration_response(self, content: str, old_package: str, new_package: str) -> Dict:
        """Structure a plain text response into the expected format"""
        return {
            "installation": {
                "remove": f"npm uninstall {old_package}",
                "add": f"npm install {new_package}"
            },
            "imports": {
                "old": f"const {old_package} = require('{old_package}')",
                "new": f"const {new_package} = require('{new_package}')"
            },
            "api_mappings": [
                {
                    "old": f"{old_package}.method()",
                    "new": f"{new_package}.method()",
                    "note": "Check documentation for exact API differences"
                }
            ],
            "examples": [
                {
                    "title": "Basic Usage",
                    "old_code": f"// Using {old_package}\n{content[:200] if len(content) > 200 else content}",
                    "new_code": f"// Using {new_package}\n// See generated migration guide"
                }
            ],
            "pitfalls": [
                {
                    "issue": "API differences",
                    "solution": "Review the new package documentation for API changes"
                }
            ]
        }
    
    def analyze_migration_complexity(self, from_package: str, to_package: str, code_usages: List[CodeUsage]) -> MigrationGuide:
        """Analyze migration complexity and generate a comprehensive guide"""
        
        prompt = f"""Analyze the complexity of migrating from '{from_package}' to '{to_package}'.

Usage count: {len(code_usages)} locations
Package types: npm packages

Provide a detailed migration analysis as JSON with:
- complexity: "low", "medium", or "high"
- estimated_time: estimated hours/days
- migration_steps: array of step-by-step instructions
- risks: array of potential risks
- benefits: array of benefits after migration
- automation_possible: boolean indicating if migration can be automated

Base your analysis on the number of usages and typical migration patterns."""

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "model": "blackboxai"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json().get('message', response.text)
                
                # Extract or create structured data
                try:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        analysis = json.loads(json_match.group())
                    else:
                        # Default structure if parsing fails
                        analysis = {
                            "complexity": "medium",
                            "estimated_time": "2-4 hours",
                            "migration_steps": ["Analyze current usage", "Install new package", "Update imports", "Test thoroughly"],
                            "risks": ["API differences", "Breaking changes"],
                            "benefits": ["Better performance", "Active maintenance"],
                            "automation_possible": True
                        }
                    
                    # Generate migration code
                    usage_examples = [usage.code_snippet for usage in code_usages[:5]]
                    code_examples = self.generate_migration_code(from_package, to_package, usage_examples)
                    
                    return MigrationGuide(
                        from_package=from_package,
                        to_package=to_package,
                        migration_steps=analysis.get('migration_steps', []),
                        code_examples=code_examples,
                        complexity=analysis.get('complexity', 'medium'),
                        estimated_time=analysis.get('estimated_time', 'Unknown')
                    )
                    
                except Exception as e:
                    logger.error(f"Error parsing migration analysis: {e}")
                    return self._default_migration_guide(from_package, to_package, code_usages)
            else:
                return self._default_migration_guide(from_package, to_package, code_usages)
                
        except Exception as e:
            logger.error(f"Migration analysis error: {e}")
            return self._default_migration_guide(from_package, to_package, code_usages)
    
    def _default_migration_guide(self, from_package: str, to_package: str, code_usages: List[CodeUsage]) -> MigrationGuide:
        """Return a default migration guide when API fails"""
        return MigrationGuide(
            from_package=from_package,
            to_package=to_package,
            migration_steps=[
                f"1. Uninstall {from_package}: npm uninstall {from_package}",
                f"2. Install {to_package}: npm install {to_package}",
                "3. Update all import statements",
                "4. Review API differences and update method calls",
                "5. Run tests to ensure functionality"
            ],
            code_examples={
                "installation": {
                    "remove": f"npm uninstall {from_package}",
                    "add": f"npm install {to_package}"
                }
            },
            complexity="medium" if len(code_usages) > 10 else "low",
            estimated_time=f"{len(code_usages) * 15} minutes" if len(code_usages) < 20 else "Several hours"
        )

# --- SERVICES / ANALYZERS ---

@st.cache_data(ttl=3600)
def get_package_info(package_name: str) -> Optional[PackageInfo]:
    """Fetch comprehensive package information from NPM."""
    session = requests.Session()
    try:
        registry_url = f"{Config.NPM_REGISTRY_BASE}/{package_name}"
        response = session.get(registry_url)
        response.raise_for_status()
        data = response.json()

        # Fetch weekly downloads
        downloads_url = f"{Config.NPM_API_BASE}/downloads/point/last-week/{package_name}"
        downloads_response = session.get(downloads_url)
        weekly_downloads = downloads_response.json().get("downloads", 0) if downloads_response.ok else 0

        latest_version = data.get("dist-tags", {}).get("latest", "")
        version_data = data.get("versions", {}).get(latest_version, {})
        time_data = data.get("time", {})
        last_publish_str = time_data.get(latest_version, time_data.get("modified"))
        
        last_publish = None
        if last_publish_str:
            try:
                last_publish = datetime.fromisoformat(last_publish_str.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Could not parse date: {last_publish_str}")

        deprecated = bool(version_data.get("deprecated", False))
        deprecated_message = version_data.get("deprecated") if isinstance(version_data.get("deprecated"), str) else None

        return PackageInfo(
            name=package_name,
            version=latest_version,
            last_publish=last_publish,
            deprecated=deprecated,
            deprecated_message=deprecated_message,
            description=data.get("description"),
            dependencies_count=len(version_data.get("dependencies", {})),
            weekly_downloads=weekly_downloads,
        )
    except requests.HTTPError as e:
        logger.error(f"Package '{package_name}' not found or API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching package info for {package_name}: {e}")
        return None

@st.cache_data(ttl=3600)
def suggest_alternatives_with_groq(package_name: str, reason: str, _groq_client) -> List[Tuple[str, str]]:
    """Use Groq to suggest alternative packages"""
    if not _groq_client:
        return []
    
    prompt = f"""Suggest 3 modern alternatives to the npm package '{package_name}'.
Reason for replacement: {reason}

For each alternative, provide:
1. Package name
2. One-line reason why it's better

Return as JSON array: [{{"name": "...", "reason": "..."}}]"""

    try:
        response = _groq_client.chat.completions.create(
            model=Config.GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": "You are an npm expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        alternatives = json.loads(content)
        return [(alt['name'], alt['reason']) for alt in alternatives[:3]]
        
    except Exception as e:
        logger.error(f"Error suggesting alternatives: {e}")
        # Fallback suggestions for common packages
        fallbacks = {
            'request': [('axios', 'Modern HTTP client with promises'), ('node-fetch', 'Lightweight fetch API')],
            'moment': [('date-fns', 'Modular and tree-shakeable'), ('dayjs', 'Lightweight with similar API')],
            'lodash': [('ramda', 'Functional programming focused'), ('lodash-es', 'ES modules version')]
        }
        return fallbacks.get(package_name, [])

@st.cache_data(ttl=3600)
def analyze_package_with_llm(package_info: PackageInfo, _groq_client) -> Dict:
    """Analyze package using Groq for a detailed report."""
    if not _groq_client:
        return {"error": "Groq client not initialized."}
    
    prompt = f"""
    Analyze the npm package based on the following data.
    - Name: {package_info.name}
    - Version: {package_info.version}
    - Last Published: {package_info.last_publish}
    - Dependencies: {package_info.dependencies_count}
    - Weekly Downloads: {package_info.weekly_downloads:,}
    - Description: {package_info.description}

    Return a single, valid JSON object with:
    - security_score: 0-10
    - maintenance_score: 0-10
    - popularity_trend: "growing", "stable", or "declining"
    - overall_health: "good", "concerning", or "critical"
    - vulnerabilities: array of up to 3 key risks
    - detailed_analysis: 300-word summary
    - needs_migration: boolean indicating if migration is recommended
    """
    try:
        response = _groq_client.chat.completions.create(
            model=Config.GROQ_DEEP_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert npm package analyst. Your response must be a single, valid JSON object without markdown or other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM analysis error: {e}")
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def scan_github_repo(repo_url: str, package_name: str) -> List[CodeUsage]:
    """Scan a public GitHub repository for package usage."""
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", repo_url)
    if not match:
        st.error("Invalid GitHub URL. Use format: https://github.com/owner/repo")
        return []

    owner, repo = match.groups()
    logger.info(f"Scanning repository: {owner}/{repo}")
    
    import_patterns = re.compile(
        f"import.*from\\s+['\"]({package_name})['\"]|"
        f"import\\s+['\"]({package_name})['\"]|"
        f"require\\(\\s*['\"]({package_name})['\"]\\s*\\)"
    )
    
    results = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Gordian-AI/1.0'})
    
    def analyze_path(path=""):
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        try:
            response = session.get(api_url)
            response.raise_for_status()
            for item in response.json():
                if item['type'] == 'dir' and item['name'] not in ['node_modules', '.git', 'dist', 'build']:
                    analyze_path(item['path'])
                elif item['type'] == 'file' and item['name'].endswith(('.js', '.jsx', '.ts', '.tsx')):
                    analyze_file_content(item)
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                logger.error(f"GitHub API rate limit hit: {e}")
                if not path:
                    st.error("GitHub API rate limit reached. Try again later.")
            elif not path:
                st.warning(f"Could not access repo. It may be private or non-existent.")

    def analyze_file_content(item: Dict):
        try:
            content_response = session.get(item['download_url'])
            content_response.raise_for_status()
            for i, line in enumerate(content_response.text.splitlines(), 1):
                if import_patterns.search(line):
                    results.append(CodeUsage(
                        file_path=item['path'],
                        line_number=i,
                        code_snippet=line.strip()
                    ))
        except Exception as e:
            logger.error(f"Could not read file {item['path']}: {e}")

    analyze_path()
    return results

# --- HELPER FUNCTIONS ---
def get_staleness_status(pkg_info: PackageInfo) -> tuple[bool, str]:
    """Determines if a package is stale and returns a status string."""
    if not pkg_info.last_publish:
        return True, "No publish date available"
    
    days_since = (datetime.now(pkg_info.last_publish.tzinfo) - pkg_info.last_publish).days
    
    if days_since > Config.VERY_STALE_DAYS_THRESHOLD:
        return True, f"🕰️ Very stale ({days_since} days ago)"
    if days_since > Config.STALE_DAYS_THRESHOLD:
        return True, f"🕰️ Stale ({days_since} days ago)"
    return False, f"✅ Active ({days_since} days ago)"

# --- UI COMPONENTS ---
def ui_sidebar():
    """Renders the sidebar for configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.groq_client:
                st.success("GROQ ✓")
            else:
                st.error("GROQ ✗")
        with col2:
            if st.session_state.blackbox_agent:
                st.success("BlackboxAI ✓")
            else:
                st.error("BlackboxAI ✗")
        
        st.text_input(
            "GitHub Repo URL",
            value="https://github.com/expressjs/express",
            key="repo_url",
            help="URL of the public GitHub repo to scan."
        )
        
        st.divider()
        st.caption("API Keys are configured via environment variables")

def ui_package_input():
    """Renders the main package input and analysis trigger."""
    st.subheader("📦 Analyze a Package")
    package_name = st.text_input("Enter NPM package name", placeholder="e.g., express, react", key="package_input")
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True) and package_name:
        st.session_state.last_analysis = None
        st.session_state.code_review = None
        st.session_state.migration_guide = None
        with st.spinner(f"Fetching data for '{package_name}'..."):
            pkg_info = get_package_info(package_name)
            if pkg_info:
                st.session_state.current_package = pkg_info
                st.success(f"Loaded **{pkg_info.name}@{pkg_info.version}**")
                
                st.markdown(f"**Description**: *{pkg_info.description or 'N/A'}*")
                is_stale, reason = get_staleness_status(pkg_info)
                if pkg_info.deprecated:
                    st.error(f"**Status**: ⚠️ DEPRECATED")
                    if pkg_info.deprecated_message:
                        st.caption(f"Reason: {pkg_info.deprecated_message}")
                elif is_stale:
                    st.warning(f"**Status**: {reason}")
                else:
                    st.success(f"**Status**: {reason}")
            else:
                st.error(f"Package '{package_name}' not found.")
                st.session_state.current_package = None

def ui_migration_section(pkg_info: PackageInfo, analysis: Dict):
    """Render the migration section with BlackboxAI integration"""
    st.markdown("### 🔄 Migration Assistant")
    
    # Check if migration is needed
    needs_migration = (
        pkg_info.deprecated or 
        analysis.get('needs_migration', False) or
        analysis.get('overall_health') == 'critical'
    )
    
    if needs_migration:
        st.warning("⚠️ This package needs migration!")
        
        # Get alternative suggestions
        reason = "deprecated" if pkg_info.deprecated else "outdated/unmaintained"
        alternatives = suggest_alternatives_with_groq(
            pkg_info.name, 
            reason, 
            st.session_state.groq_client
        )
        
        if alternatives:
            st.markdown("#### Recommended Alternatives:")
            
            selected_alternative = None
            for alt_name, alt_reason in alternatives:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{alt_name}** - {alt_reason}")
                with col2:
                    if st.button(f"Migrate to {alt_name}", key=f"migrate_{alt_name}"):
                        selected_alternative = alt_name
            
            # Generate migration guide if alternative selected
            if selected_alternative or st.session_state.get('selected_alternative'):
                if selected_alternative:
                    st.session_state.selected_alternative = selected_alternative
                
                target_package = st.session_state.selected_alternative
                
                with st.spinner(f"🤖 Generating migration guide from {pkg_info.name} to {target_package}..."):
                    # Get code usages if available
                    code_usages = st.session_state.get('code_review', {}).get('usages', [])
                    
                    # Generate migration guide
                    if st.session_state.blackbox_agent:
                        migration_guide = st.session_state.blackbox_agent.analyze_migration_complexity(
                            pkg_info.name,
                            target_package,
                            code_usages
                        )
                        st.session_state.migration_guide = migration_guide
                        
                        # Display migration guide
                        st.markdown(f"### 📋 Migration Guide: `{pkg_info.name}` → `{target_package}`")
                        
                        # Complexity and time estimate
                        complexity_color = {
                            "low": "🟢",
                            "medium": "🟡", 
                            "high": "🔴"
                        }.get(migration_guide.complexity, "⚪")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Complexity", f"{complexity_color} {migration_guide.complexity.title()}")
                        with col2:
                            st.metric("Estimated Time", migration_guide.estimated_time)
                        
                        # Migration steps
                        with st.expander("📝 Migration Steps", expanded=True):
                            for i, step in enumerate(migration_guide.migration_steps, 1):
                                st.markdown(f"{i}. {step}")
                        
                        # Code examples
                        if isinstance(migration_guide.code_examples, dict):
                            with st.expander("💻 Code Examples", expanded=True):
                                if 'installation' in migration_guide.code_examples:
                                    st.markdown("**Installation:**")
                                    install = migration_guide.code_examples['installation']
                                    st.code(f"# Remove old package\n{install.get('remove', '')}\n\n# Install new package\n{install.get('add', '')}", language="bash")
                                
                                if 'imports' in migration_guide.code_examples:
                                    st.markdown("**Import Changes:**")
                                    imports = migration_guide.code_examples['imports']
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("❌ Old:")
                                        st.code(imports.get('old', ''), language="javascript")
                                    with col2:
                                        st.markdown("✅ New:")
                                        st.code(imports.get('new', ''), language="javascript")
                                
                                if 'api_mappings' in migration_guide.code_examples:
                                    st.markdown("**API Changes:**")
                                    for mapping in migration_guide.code_examples.get('api_mappings', []):
                                        st.markdown(f"- `{mapping['old']}` → `{mapping['new']}`")
                                        if mapping.get('note'):
                                            st.caption(f"  💡 {mapping['note']}")
                                
                                if 'examples' in migration_guide.code_examples:
                                    st.markdown("**Complete Examples:**")
                                    for example in migration_guide.code_examples.get('examples', []):
                                        st.markdown(f"**{example.get('title', 'Example')}:**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("❌ Old Code:")
                                            st.code(example.get('old_code', ''), language="javascript")
                                        with col2:
                                            st.markdown("✅ New Code:")
                                            st.code(example.get('new_code', ''), language="javascript")
                                
                                if 'pitfalls' in migration_guide.code_examples:
                                    st.markdown("**⚠️ Common Pitfalls:**")
                                    for pitfall in migration_guide.code_examples.get('pitfalls', []):
                                        st.warning(f"**Issue:** {pitfall['issue']}\n\n**Solution:** {pitfall['solution']}")
                    else:
                        st.error("BlackboxAI not configured. Cannot generate migration guide.")
    else:
        st.success("✅ This package is healthy and doesn't require immediate migration.")

def ui_dashboard(pkg_info: PackageInfo):
    """Renders the main dashboard with tabs."""
    st.header(f"Analytics Dashboard: `{pkg_info.name}`")

    # --- Action Buttons ---
    col1, col2, col3 = st.columns(3)
    if col1.button("🤖 Run AI Analysis", use_container_width=True):
        with st.spinner("🧠 Performing deep analysis with AI..."):
            st.session_state.last_analysis = analyze_package_with_llm(pkg_info, st.session_state.groq_client)
    
    if col2.button("📝 Review Code Usage", use_container_width=True):
        with st.spinner(f"Scanning {st.session_state.repo_url} for `{pkg_info.name}` usage..."):
            usages = scan_github_repo(st.session_state.repo_url, pkg_info.name)
            st.session_state.code_review = {'usages': usages} if usages else None
            if usages:
                st.success(f"Found {len(usages)} usage locations.")
            else:
                st.warning(f"No usage of `{pkg_info.name}` found.")
    
    if col3.button("🔄 Check Migration", use_container_width=True):
        # This will trigger the migration section to appear
        st.session_state.show_migration = True
    
    st.divider()

    # --- Dashboard Tabs ---
    tabs = st.tabs(["📊 Overview", "🛡️ Vulnerabilities", "📈 Popularity", "📝 Code Review", "🔄 Migration"])

    analysis = st.session_state.get('last_analysis', {})
    
    with tabs[0]:  # Overview
        if analysis and "error" not in analysis:
            health_map = {"good": "🟢", "concerning": "🟡", "critical": "🔴"}
            health = analysis.get('overall_health', 'N/A')
            st.metric("Overall Health", f"{health_map.get(health, '⚪️')} {health.title()}")
            st.markdown("##### AI Summary")
            st.info(analysis.get('detailed_analysis', 'Not available.'))
            st.metric("Maintenance Score", f"{analysis.get('maintenance_score', 'N/A')}/10")
        else:
            st.info("Click 'Run AI Analysis' to generate an AI-powered overview.")

    with tabs[1]:  # Vulnerabilities
        if analysis and "error" not in analysis:
            st.metric("Security Score", f"{analysis.get('security_score', 'N/A')}/10")
            st.markdown("##### Potential Vulnerabilities & Risks")
            for risk in analysis.get('vulnerabilities', []):
                st.warning(f"**-** {risk}")
        else:
            st.info("Run AI analysis to see vulnerability information.")

    with tabs[2]:  # Popularity
        col1, col2 = st.columns(2)
        col1.metric("Weekly Downloads", f"{pkg_info.weekly_downloads:,}")
        if analysis and "error" not in analysis:
            trend_map = {"growing": "📈", "stable": "➡️", "declining": "📉"}
            trend = analysis.get('popularity_trend', 'Unknown')
            col2.metric("Popularity Trend", f"{trend_map.get(trend, '❓')} {trend.title()}")
        else:
            col2.metric("Popularity Trend", "❓ Unknown")

    with tabs[3]:  # Code Review
        code_review = st.session_state.get('code_review')
        if code_review:
            st.metric("Code Usage Locations", len(code_review['usages']))
            with st.expander(f"View {len(code_review['usages'])} usage locations"):
                for usage in code_review['usages'][:20]:  # Limit display
                    st.code(f"# {usage.file_path}:{usage.line_number}\n{usage.code_snippet}", language="javascript")
        else:
            st.info("Click 'Review Code Usage' to scan the configured GitHub repo.")

    with tabs[4]:  # Migration
        if st.session_state.get('show_migration') or pkg_info.deprecated:
            ui_migration_section(pkg_info, analysis)
        else:
            st.info("Click 'Check Migration' to see migration options and generate migration guides.")


def main():
    """Main Streamlit application entrypoint."""
    st.set_page_config(page_title="Gordian AI", page_icon="🛡️", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stMetric {
            border: 1px solid #2e2e2e;
            border-radius: 10px;
            padding: 10px;
            background-color: #0F1116;
        }
        .migration-card {
            background-color: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 10px 0;
        }
        code {
            background-color: #2d2d2d;
            padding: 2px 4px;
            border-radius: 3px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ Gordian - AI Driven Software Intelligence")
    st.caption("Powered by GROQ and BlackboxAI for intelligent package analysis and migration assistance")

    # Initialize state
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = Groq(api_key=Config.GROQ_API_KEY) if Groq and Config.GROQ_API_KEY else None
    
    if 'blackbox_agent' not in st.session_state:
        st.session_state.blackbox_agent = BlackboxAIAgent(Config.BLACKBOX_API_KEY) if Config.BLACKBOX_API_KEY else None

    # Render UI
    ui_sidebar()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ui_package_input()
    with col2:
        if 'current_package' in st.session_state and st.session_state.current_package:
            ui_dashboard(st.session_state.current_package)
        else:
            # Welcome message with features
            st.markdown("""
            ### 👈 Enter a package name to begin
            
            **Features:**
            - 🤖 **AI Analysis** - Deep package health assessment using GROQ
            - 📝 **Code Review** - Scan GitHub repos for package usage
            - 🔄 **Migration Assistant** - Generate complete migration guides with BlackboxAI
            - 💻 **Code Generation** - Get working migration code examples
            - ⚡ **Smart Suggestions** - AI-powered alternative package recommendations
            
            **How it works:**
            1. Enter an npm package name
            2. Run AI analysis to assess package health
            3. Review code usage in your repository
            4. Generate migration guides if needed
            5. Get complete, working code examples for migration
            """)
            

if __name__ == "__main__":
    main()
