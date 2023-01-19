# this file imports custom routes into the experiment server

from flask import Blueprint, render_template, request, jsonify, Response, abort, current_app
from jinja2 import TemplateNotFound
from functools import wraps
from sqlalchemy import or_

from psiturk.psiturk_config import PsiturkConfig
from psiturk.experiment_errors import ExperimentError, InvalidUsage
from psiturk.user_utils import PsiTurkAuthorization, nocache

from psiturk.psiturk_statuses import ALLOCATED, BONUSED, COMPLETED, \
    CREDITED, NOT_ACCEPTED, QUITEARLY, STARTED, SUBMITTED
IGNORE = 100 #custom status for ignoring data for conditioning purposes
PSITURK_STATUSES = {
    "ALLOCATED": ALLOCATED,
    "BONUSED": BONUSED,
    "COMPLETED": COMPLETED,
    "CREDITED": CREDITED,
    "NOT_ACCEPTED": NOT_ACCEPTED,
    "QUITEARLY": QUITEARLY,
    "STARTED": STARTED,
    "SUBMITTED": SUBMITTED,
    "IGNORE": IGNORE
}
import datetime
import json
from collections import Counter

# # Database setup
from psiturk.db import db_session, init_db
from psiturk.models import Participant
from json import dumps, loads

# load the configuration options
config = PsiturkConfig()
config.load_config()
myauth = PsiTurkAuthorization(config)  # if you want to add a password protect route use this

# explore the Blueprint
custom_code = Blueprint('custom_code', __name__, template_folder='templates', static_folder='static')

def get_participants(codeversion):
    return (
        Participant
        .query
        .filter(Participant.codeversion == codeversion)
        .all()
    )

@custom_code.route("/set_codeversion", methods=["GET"])
def set_codeversion():
    """
    Set the codeversion here rather than through the
    psiturk mechanism for more flexibility.

    E.g.,
    fetch('/set_codeversion?codeversion=test&uniqueId=debugWOtcm:debugLi689')
    """
    codeversion = request.args.get("codeversion", default=None, type=str)
    uniqueId = request.args['uniqueId']
    user = Participant.query.\
        filter(Participant.uniqueid == uniqueId).one()
    user.codeversion = codeversion
    db_session.commit()
    current_app.logger.info(f"Set {uniqueId} codeversion to {codeversion}")
    return jsonify(msg='success')

def get_condition_counts(
    n_conditions,
    cutoff_time_min,
    mode,
    experiment_code_version
):
    # Calc a starttime for participants who are still incomplete
    cutofftime = datetime.timedelta(minutes=-cutoff_time_min)
    starttime = datetime.datetime.now(datetime.timezone.utc) + cutofftime

    # Figure out the current participant counts
    participants = Participant.query.\
        filter(Participant.mode == mode).\
        filter(Participant.codeversion == experiment_code_version).\
        filter(or_(Participant.status == COMPLETED,
                   Participant.status == CREDITED,
                   Participant.status == SUBMITTED,
                   Participant.status == BONUSED,
                   Participant.beginhit > starttime)).\
        filter(Participant.status != IGNORE).all()
    cur_counts = [0 for _ in range(n_conditions)]
    for p in participants:
        cur_counts[p.cond] += 1
    return cur_counts

def generate_condition(
    n_conditions,
    cutoff_time_min,
    mode,
    experiment_code_version
):
    import random
    params = dict(
        n_conditions=n_conditions,
        cutoff_time_min=cutoff_time_min,
        mode=mode,
        experiment_code_version=experiment_code_version
    )
    cur_counts = get_condition_counts(
        n_conditions=n_conditions,
        cutoff_time_min=cutoff_time_min,
        mode=mode,
        experiment_code_version=experiment_code_version
    )
    min_n = min(cur_counts)
    minima = [i for i, n in enumerate(cur_counts) if n == min_n]
    return random.choice(minima)

@custom_code.route("/get_condition", methods=["GET"])
def get_condition():
    """
    Get a condition for participant.

    Get the condition here,
    rather than relying on psiturk's
    built-in conditioning mechanism. Requires that
    `n_conditions`, `cutoff_time_min`,
    `mode`, and a valid `uniqueId` be sent.

    Example:

    ```javascript
    fetch(
        '/get_condition?' +
        'n_conditions=2' +
        '&cutoff_time_min=20' +
        '&mode=live' +
        '&uniqueId=debugWOtcm:debugLi689' +
        '&experiment_code_version=default_codeversion'
    ).then(r => r.json())
    .then(console.log)
    ```
    """

    n_conditions = request.args.get("n_conditions", default=0, type=int)
    cutoff_time_min = request.args.get("cutoff_time_min", default=None, type=int)
    mode = request.args.get("mode", default='live', type=str)
    experiment_code_version = request.args.get("experiment_code_version", type=str)

    #default to config settings
    if cutoff_time_min is None:
        cutoff_time_min = config.getint('Task Parameters', 'cutoff_time')

    cond = generate_condition(
        n_conditions=n_conditions,
        cutoff_time_min=cutoff_time_min,
        mode=mode,
        experiment_code_version=experiment_code_version
    )

    uniqueId = request.args['uniqueId']
    user = Participant.query.\
        filter(Participant.uniqueid == uniqueId).one()
    user.cond = cond
    db_session.commit()
    cur_counts = get_condition_counts(
        n_conditions=n_conditions,
        cutoff_time_min=cutoff_time_min,
        mode=mode,
        experiment_code_version=experiment_code_version
    )
    condition_counts = dict(zip(range(len(cur_counts)), cur_counts))
    current_app.logger.info(f"Set {uniqueId} condition to {cond}")
    return jsonify(condition=cond, condition_counts=condition_counts)

@custom_code.route("/set_status", methods=['GET'])
def set_status():
    """ Set the worker's status """
    if 'uniqueId' not in request.args:
        resp = {"status": "Bad Request: uniqueId is required"}
        return jsonify(**resp)
    elif 'new_status' not in request.args:
        resp = {"status": "Bad Request: new_status is required"}
        return jsonify(**resp)
    unique_id = request.args['uniqueId']
    new_status = request.args['new_status'].upper()
    if new_status not in PSITURK_STATUSES:
        resp = {"status": "unrecognized new_status"}
        return jsonify(**resp)
    new_status = PSITURK_STATUSES[new_status]

    try:
        user = Participant.query.\
            filter(Participant.uniqueid == unique_id).one()
        old_status = user.status
        user.status = new_status
        db_session.add(user)
        db_session.commit()
        status_msg = f"Success: Setting {unique_id} status from {old_status} to {new_status}"
        current_app.logger.info(status_msg)
    except Exception as e:
        status_msg = str(e)
        current_app.logger.info(status_msg)
    resp = {"status": status_msg}
    return jsonify(**resp)

@custom_code.route('/compute_bonus', methods=['GET'])
def compute_bonus():
    # check that user provided the correct keys
    # errors will not be that gracefull here if being
    # accessed by the Javascrip client
    if 'uniqueId' not in request.args:
        raise ExperimentError('improper_inputs')  # i don't like returning HTML to JSON requests...  maybe should change this
    uniqueId = request.args['uniqueId']

    try:
        # lookup user in database
        user = Participant.query.\
               filter(Participant.uniqueid == uniqueId).\
               one()

        bonus = 0
        user_data = loads(user.datastring) # load datastring from JSON
        for qname, resp in user_data['questiondata'].items():
            if qname == 'bonusDollars':
                bonus = round(resp, 2)
                break
        bonus = min(bonus, 10)

        user.bonus = bonus
        db_session.add(user)
        db_session.commit()
        resp = {"bonusComputed": "success", 'bonus': bonus}
        return jsonify(**resp)
    except:
        abort(404)  # again, bad to display HTML, but...

@custom_code.route('/testexperiment')
def testexperiment():
    data = {
        key: "{{ " + key + " }}"
        for key in ['uniqueId', 'condition', 'counterbalance', 'adServerLoc', 'mode']
    }
    return render_template('exp.html', **data)

@custom_code.route('/data/<codeversion>/<name>', methods=['GET'])
@myauth.requires_auth
@nocache
def download_datafiles(codeversion, name):
    contents = {
        "trialdata": lambda p: p.get_trial_data(),
        "eventdata": lambda p: p.get_event_data(),
        "questiondata": lambda p: p.get_question_data(),
        "bonusdata": lambda p: f"{p.uniqueid},{p.bonus}\n",
        "conditiondata": lambda p: f"{p.uniqueid},{p.cond}\n"
    }

    if name not in contents:
        abort(404)

    query = get_participants(codeversion)

    # current_app.logger.critical('data %s', data)
    def ret():
        for p in query:
            try:
                yield contents[name](p)
            except TypeError:
                current_app.logger.error("Error loading {} for {}".format(name, p))
                current_app.logger.error(format_exc())
    response = Response(
        ret(),
        mimetype="text/csv",
        headers={
            'Content-Disposition': 'attachment;filename=%s.csv' % name
        })

    return response
