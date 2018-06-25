#!/usr/bin/env python
#
# (c) Copyright 2015-2017 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Tool for gathering job data from a Jenkins server and generating summary
stat reports and graphs. These reports cover metrics such as success/failure
rates and job duration.

The job data is persisted in a file as JSON lines (JSONL) (see
https://en.wikipedia.org/wiki/JSON_Streaming#Line_delimited_JSON for more
details).

The script can be run as follows,

./get_jenkins_stats.py -s <jenkins url> -j <jenkins job name> -o <output directory>

and

 cat <output directory>/<jenkins job name>.json | jq

to make JSONL readable for debugging

"""

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import jinja2
import pandas as pd
import plotly
from plotly import graph_objs as go

LOCK_RETRIES = 10
RETRY_SLEEP_SEC = 5

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    script_name = __file__
    parser.set_defaults(
        script_dir=os.path.abspath((os.path.dirname(script_name))))
    parser.add_argument('-s', '--server', dest='jenkins_url',
                        default=os.getenv('JENKINS_URL', ''),
                        help='Jenkins server URL (or set JENKINS_URL environment variable)',
                        required=True)
    parser.add_argument('-p', '--project', dest='jenkins_project',
                        default=os.getenv('JENKINS_PROJECT', ''),
                        help='Jenkins project (or set JENKINS_PROJECT environment variable)')
    parser.add_argument('-j', '--job', dest='jenkins_job',
                        default=os.getenv('JENKINS_JOB', ''),
                        help='Jenkins job (or set JENKINS_JOB environment variable)')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        default='/var/www/html/jenkins/stats/',
                        help='Path to output files to (default: %(default)s).')
    parser.add_argument('-r', '--report-range-hours', dest='range_hours',
                        type=int, default=336,
                        help='Range for report in hours (default: %(default)s).')
    parser.add_argument('-t', '--html-template', dest='html_template',
                        default='jenkins_stats.html',
                        help='Jinja2 template to use for html reports '
                             '(default: %(default)s).')
    parser.add_argument('-v', '--verbose', dest='log_verbosely',
                        action='store_true',
                        help='Show DEBUG level log output.')
    parser.add_argument('-q', '--quiet', dest='log_quietly',
                        action='store_true',
                        help='Show only ERROR log messages.')
    parser.add_argument('-l', '--logfile',
                        help='''
                        Name of the file to log messages to
                        (default: %(default)s).
                        ''',
                        default='%s.log' %
                                os.path.splitext(os.path.basename(script_name))[0])
    parser.add_argument('--no-log', dest='no_logfile', action='store_true',
                        help='Do not write log output to file.')
    args = parser.parse_args()
    configure_logging(args)

    # enable logging multiple columns in output
    pd.set_option('display.width', 1000)

    dir_path = os.path.join(args.output_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    global lock_file

    finish_dt = datetime.now()
    start_dt = finish_dt - timedelta(hours=args.range_hours)
    log.info('Reporting on jobs between %s and %s',
             start_dt.strftime('%H:%M:%S %d-%b-%Y'),
             finish_dt.strftime('%H:%M:%S %d-%b-%Y'))

    projects = {}
    for project in get_projects(args):
        project_data_dir = os.path.join(dir_path, project)
        projects[project] = {}
        if not os.path.exists(project_data_dir):
            os.makedirs(project_data_dir)
        for branch in get_branches(args, project):
            data_file = os.path.join(project_data_dir, '%s.json' % branch)
            # lock the data-file before we do anything else, since we don't want
            # another writer modifying the file after we've read but before we've
            # written (this is all redundant if we use a db)
            create_lock(data_file)
            projects[project][branch] = get_runs(args, data_file, project, branch)

    df_builds = projects_to_dataframe(projects)
    df_overall_stats = generate_overall_build_stats(args, df_builds, start_dt)

    for project in get_projects(args):
        project_data_dir = os.path.join(dir_path, project)
        for branch in get_branches(args, project):
            html = generate_html(args, df_overall_stats)
            write_html(branch, project_data_dir, html)


def create_lock(data_file):
    global lock_file
    lock_file = '%s.lck' % data_file
    log.debug('Using lock file %s', lock_file)
    if os.path.exists(lock_file):
        lock = open(lock_file, 'r+')
    else:
        lock = open(lock_file, 'w')
    for attempt in range(1, LOCK_RETRIES + 1):
        log.debug('Locking attempt %d of %d', attempt, LOCK_RETRIES)
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            log.debug('%s already locked, sleep and retry', lock_file)
            time.sleep(RETRY_SLEEP_SEC)
        else:
            log.debug('%s locked, continuing', lock_file)
            # we have the lock, we're done
            break
    else:
        log.critical('Failed to lock file for write, aborting')
        exit(1)


def write_html(branch, dir_path, html):
    file_name = '%s.html' % branch
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w') as f:
        f.write(html)
    log.info('Wrote %s', file_path)


def generate_html(args, df_overall_stats):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader('.'),
        autoescape=False
    )
    template = env.get_template(args.html_template)
    report_units = '%g days' % (args.range_hours / 24)
    if args.range_hours <= 24:
        report_units = '%d hours' % args.range_hours
    html = template.render(
        title='%s for last %s' % (args.jenkins_job, report_units),
        status_plot=plot_status(df_overall_stats),
        duration_plot=plot_duration(df_overall_stats),
        timestamp_dt=datetime.now(),
        df_stats=df_overall_stats)
    return html


def generate_overall_build_stats(args, df, start_dt):
    log.debug('Generating overall stats')

    # resample data for plots
    sample_window = '1D'
    if args.range_hours <= 24:
        sample_window = '1H'
    df_stats = pd.DataFrame()
    # ignoring aborts from total and pct calc
    df_stats['success'] = df.success.resample(sample_window).sum()
    df_stats['failure'] = df.failure.resample(sample_window).sum()
    df_stats['aborted'] = df.aborted.resample(sample_window).sum()
    df_stats['failure_infr'] = df.failure_infr.resample(sample_window).sum()
    df_stats['failure_ours'] = df.failure_ours.resample(sample_window).sum()
    df_stats['failure_other'] = df.failure_other.resample(sample_window).sum()
    df_stats['total'] = df_stats.success + df_stats.failure
    df_stats['success_pct'] = df_stats.success / df_stats.total * 100
    df_stats['failure_pct'] = df_stats.failure / df_stats.total * 100
    df_stats['failure_infr_pct'] = df_stats.failure_infr / df_stats.total * 100
    df_stats['failure_ours_pct'] = df_stats.failure_ours / df_stats.total * 100
    df_stats['failure_other_pct'] = df_stats.failure_other / df_stats.total * 100
    df_stats['duration_sec_min'] = df.duration_sec.resample(sample_window).min()
    df_stats['duration_sec_max'] = df.duration_sec.resample(sample_window).max()
    df_stats['duration_sec_avg'] = df.duration_sec.resample(
        sample_window).mean()
    df_success = df[df['success']]
    df_stats['success_duration_min_avg'] = df_success.duration_sec.resample(
        sample_window).mean() / 60
    # resample doesn't currently support percentile
    # (https://github.com/pandas-dev/pandas/issues/15023)
    df_stats['success_duration_min_90th'] = df_success.duration_sec.groupby(
        pd.Grouper(freq=sample_window)).quantile(0.9) / 60
    df_stats['success_duration_min_50th'] = df_success.duration_sec.groupby(
        pd.Grouper(freq=sample_window)).quantile(0.5) / 60
    df_stats = df_stats.round(decimals=1)

    # restrict to builds since start_dt
    if start_dt is not None:
        df_stats = df_stats[df_stats.index > start_dt]

    df_stats.fillna(value=0, inplace=True)
    log.debug('df_stats:\n%s\n', df_stats)
    return df_stats


def projects_to_dataframe(projects):
    """
    Convert build data to pandas dataframe for subsequent analysis
    """

    build_data = dict()
    build_data['timestamp'] = list()
    build_data['project'] = list()
    build_data['branch'] = list()
    build_data['success'] = list()
    build_data['failure'] = list()
    build_data['aborted'] = list()
    build_data['failure_node'] = list()
    build_data['failure_infr'] = list()
    build_data['failure_ours'] = list()
    build_data['failure_other'] = list()
    build_data['duration_sec'] = list()

    for project_name, project in projects.items():
        for branch_name, branch in project.items():
            for number, build in branch.items():
                # time, success, failure, aborted, duration
                success, failure, aborted = False, False, False
                if build['result'] == 'SUCCESS':
                    success = True
                elif build['result'] == 'FAILURE':
                    failure = True
                elif build['result'] == 'ABORTED':
                    aborted = True
                elif build['result'] in ('NOT_BUILT', 'UNKNOWN'):
                    continue
                else:
                    log.critical('Unknown status on project %s, branch %s, run %s: %s',
                                 project_name, branch_name, number, build['result'])
                    exit(1)

                build_data['timestamp'].append(build['start_time'])
                build_data['project'].append(build['project'])
                build_data['branch'].append(build['branch'])
                build_data['success'].append(success)
                build_data['failure'].append(failure)
                build_data['aborted'].append(aborted)
                build_data['failure_node'].append(build['failed_at'])
                build_data['duration_sec'].append(build['duration_sec'])

                infr, ours, other = False, False, False
                override = failure_overrides(project, branch, number)
                if (build['failed_at'] is None
                        or override == 'other'):
                    if failure:
                        other = True
                elif (build['failed_at'] in ()
                      or build['failed_at'].startswith('Deploy -')
                      or build['failed_at'].startswith('Promote -')
                      or build['failed_at'].startswith('Build Infrastructure -')
                      or override == 'infr'):
                    infr = True
                elif (build['failed_at'] in ('Build', 'Test', 'Sonar Scan', 'Security Checks')
                      or build['failed_at'].startswith('Smoke Test -')
                      or build['failed_at'].startswith('Functional Test -')
                      or override == 'ours'):
                    ours = True
                else:
                    log.critical('Unknown run node on project %s, branch %s, run %s: %s',
                                 project_name, branch_name, number, build['failed_at'])
                    exit(1)
                build_data['failure_infr'].append(infr)
                build_data['failure_ours'].append(ours)
                build_data['failure_other'].append(other)

    df = pd.DataFrame(
        build_data,
        columns=['timestamp', 'project', 'branch', 'success', 'failure', 'aborted', 'failure_node', 'failure_infr',
                 'failure_ours', 'failure_other', 'duration_sec'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    del df['timestamp']
    log.debug('df:\n%s\n', df)
    return df


override_file = None


def failure_overrides(project, branch, run):
    return None


class Result:
    ok = None
    file = None

    def json(self):
        return json.load(open(self.file[1]))


class JenkinsGet:

    def get(self, url, params=None, verify=None):
        from tempfile import mkstemp
        import subprocess
        r = Result()
        r.file = mkstemp(suffix='.json')
        r.ok = False
        param_str = ''

        if params:
            param_str = '?'
            for i, j in params.items():
                param_str += '%s=%s&' % (i, j)
            param_str = param_str[:-1]

        result = subprocess.check_call(['wget', '--no-verbose', '--load-cookies', 'cookies.txt',
                                        '--output-document', str(r.file[1]),
                                        str(url) + param_str])
        if result == 0:
            r.ok = True
        return r


def get_projects(args):
    """Return a list of valid projects for the server"""
    # https://build.platform.hmcts.net/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/?tree=name
    payload = {'tree': 'name'}
    jenkins_url = '%s/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/' % (args.jenkins_url)
    log.debug('Retrieving projects from %s', jenkins_url)
    session = JenkinsGet()
    r = session.get(jenkins_url, params=payload)
    if not r.ok:
        log.critical("Failed to fetch from %s", jenkins_url)
        exit(1)
    jenkins_data = r.json()
    projects = list()
    for project in jenkins_data:
        if project['name'].startswith(args.jenkins_project):
            projects.append(project['name'])
    return projects


def get_branches(args, project):
    """Return a list of valid jobs for a project"""
    # https://build.platform.hmcts.net/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/ccd-data-store-api/branches/?tree=name
    payload = {'tree': 'name'}
    jenkins_url = '%s/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/%s/branches/' % (
    args.jenkins_url, project)
    log.debug('Retrieving jobs from %s', jenkins_url)
    session = JenkinsGet()
    r = session.get(jenkins_url, params=payload)
    if not r.ok:
        log.critical("Failed to fetch from %s", jenkins_url)
        exit(1)
    jenkins_data = r.json()
    branches = []
    for branch in jenkins_data:
        if branch['name'].lower().startswith(args.jenkins_job):
            branches.append(branch['name'])
    return branches


def get_runs(args, data_file, project, branch):
    # read from data_file if it exists, create new one if it doesn't
    builds = dict()

    first_run = False
    try:
        with open(data_file, 'r') as f:
            for line in f:
                build = json.loads(line)
                number = build['run']
                log.debug('Read existing record for build %s', number)
                builds[number] = build
    except (IOError, EOFError):
        first_run = True
        log.warning('%s not found, creating new data_file', data_file)

    log.info('Read %d builds from %s', len(builds), data_file)

    # https://build.platform.hmcts.net/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/ccd-admin-web/branches/master/runs/
    jenkins_url = '%s/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/%s/branches/%s/' % (
        args.jenkins_url, project, branch)
    payload = {'tree': 'latestRun[id]'}
    if first_run:
        # If this is the first run for a job, pull all builds from Jenkins,
        # otherwise use the standard jenkins call which limits to 100 results
        # (the assumption is we're running regularly enough after the first run
        # that those 100 results is sufficient)
        log.info('%s is a new job, querying all builds', job)
        payload = {'tree': 'allBuilds[number]'}

    log.debug('Retrieving jenkins data from %s', jenkins_url)
    session = JenkinsGet()
    r = session.get(jenkins_url, params=payload)
    if not r.ok:
        log.critical("Failed to fetch from %s", jenkins_url)
        exit(1)
    jenkins_data = r.json()
    new_builds = list()

    for run in range(int(jenkins_data['latestRun']['id']), 0, -1):
        if run in builds:
            log.debug('Already stored record for run %d', run)
            continue
        log.debug('Retrieving record for run %d', run)
        jenkins_url = '%s/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/%s/branches/%s/runs/%s/' % (
            args.jenkins_url, project, branch, run)
        r = session.get(jenkins_url)
        build_data = r.json()
        result = build_data['result']
        duration = build_data['durationInMillis']
        if result is None or duration == 0:
            log.debug(
                'Skipping unfinished run %s (result = %s, duration = %d)',
                run, result, duration)
            continue

        change_hash = build_data['commitId']

        build_start_time = build_data['startTime']
        build_duration_sec = int(build_data['durationInMillis'] / 1000)
        build_end_time = build_data['endTime']

        failed_at = None
        if result == 'FAILURE':
            jenkins_url = '%s/blue/rest/organizations/jenkins/pipelines/HMCTS/pipelines/%s/branches/%s/runs/%s/nodes/' % (
                args.jenkins_url, project, branch, run)
            payload = {'tree': 'displayName,result'}
            r = session.get(jenkins_url, payload)
            for node in r.json():
                if node['result'] == 'FAILURE':
                    failed_at = node['displayName']

        build = {'run': run,
                 'project': project,
                 'branch': branch,
                 'change_hash': change_hash,
                 'result': result,
                 'failed_at': failed_at,
                 'start_time': build_start_time,
                 'end_time': build_end_time,
                 'duration_sec': build_duration_sec,
                 }
        builds[run] = build
        new_builds.append(build)
    if len(new_builds) > 0:
        with open(data_file, 'a') as f:
            for build in new_builds:
                log.info('Storing new record for run %s', build['run'])
                json.dump(build, f, separators=(',', ':'))
                f.write('\n')
            log.debug('Wrote %d new results to data_file %s',
                      len(new_builds), data_file)
    else:
        log.info('No new jobs to write to %s', data_file)

    return builds


def find_action(actions, action_type):
    """Extract the 'action_type' from an build['actions']"""
    found_actions = []
    for action in actions:
        if '_class' in action and action['_class'] == action_type:
            found_actions.append(action)
    return found_actions


def plot_status(df):
    plot_title = 'Success/Failure rates'

    # colours from http://clrs.cc/
    success = go.Bar(
        x=df.index,
        y=df['success_pct'],
        name='% Success',
        marker=dict(
            color='#2ca02c'
        )
    )
    failure_infr = go.Bar(
        x=df.index,
        y=df['failure_infr_pct'],
        name='% Failure Infrastructure',
        marker=dict(
            color='#de3c0c'
        )
    )
    failure_ours = go.Bar(
        x=df.index,
        y=df['failure_ours_pct'],
        name='% Failure Ours',
        marker=dict(
            color='#ed960b'
        )
    )
    failure_other = go.Bar(
        x=df.index,
        y=df['failure_other_pct'],
        name='% Failure Unknown',
        marker=dict(
            color='#dddddd'
        )
    )
    jobs = go.Scatter(
        x=df.index,
        y=df['total'],
        name='Jobs',
        yaxis='y2',
        marker=dict(
            color='#1b689d'
        )
    )
    data = [success, failure_infr, failure_ours, failure_other, jobs]

    layout = go.Layout(
        barmode='stack',
        title=plot_title,
        xaxis=dict(tickformat="%d-%b-%Y", tickmode="linear"),
        yaxis=dict(ticksuffix="%", range=[0, 100]),
        yaxis2=dict(title="Runs", overlaying='y', side='right', range=[0, 100]),
        legend=dict(orientation="h", x=0.02, y=1.15)
    )
    fig = go.Figure(data=data, layout=layout)
    plot = plotly.offline.plot(fig,
                               show_link=False,
                               output_type='div',
                               include_plotlyjs='False')
    return plot


def plot_duration(df):
    plot_title = 'Duration for successful jobs'

    percentile90 = go.Scatter(
        x=df.index,
        y=df['success_duration_min_90th'],
        mode='lines+markers',
        name='Duration (90th percentile)')
    percentile50 = go.Scatter(
        x=df.index,
        y=df['success_duration_min_50th'],
        mode='lines+markers',
        name='Duration (50th percentile)')

    data = [percentile50, percentile90]
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(tickformat="%d-%b-%Y", tickmode="linear"),
        yaxis=dict(ticksuffix=" min", range=[0, 40]),
        legend=dict(orientation="h", x=0.02, y=1.15)
    )
    fig = go.Figure(data=data, layout=layout)
    plot = plotly.offline.plot(fig,
                               show_link=False,
                               output_type='div',
                               include_plotlyjs='False')
    return plot


def configure_logging(args):
    """Configure logging.

    - Default => INFO
    - log_quietly (-q) => ERROR
    - log_verbosely (-v) => DEBUG
    """

    # requests and urllib3 are very chatty by default, suppress some of this
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # and suppress InsecurePlatformWarning from urllib3 also
    # see http://stackoverflow.com/questions/29099404 for details
    import requests.packages.urllib3
    requests.packages.urllib3.disable_warnings()

    # or maybe https://stackoverflow.com/a/28002687
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Set root logger level to DEBUG, and use the
    # handler level to control verbosity.
    logging.getLogger().setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stderr)
    if args.log_quietly:
        ch.setLevel(logging.ERROR)
    elif args.log_verbosely:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    ch_format = logging.Formatter('%(message)s')
    ch.setFormatter(ch_format)
    logging.getLogger().addHandler(ch)

    if not args.no_logfile:
        fh = logging.FileHandler(args.logfile, delay=True)
        # if args.log_verbosely:
        #     fh.setLevel(logging.DEBUG)
        # else:
        #     fh.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        log_format = (
            '%(asctime)s: %(process)d:%(thread)d %(levelname)s - %(message)s'
        )
        fh_format = logging.Formatter(log_format)
        fh.setFormatter(fh_format)
        logging.getLogger().addHandler(fh)


if __name__ == "__main__":
    main()
