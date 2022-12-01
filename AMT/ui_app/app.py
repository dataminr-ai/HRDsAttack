from flask import Blueprint, Flask
from flask import render_template, url_for

bp = Blueprint('burritos', __name__,
               template_folder='templates')

@bp.route("/")
def index_page():
    return render_template("Please go to testing pages.")


@bp.route("/qualification")
def render_qualification_interface():
    return render_template('qualification_interface.html')


@bp.route("/full-task")
def render_full_task():
    return render_template('full_annotation_interface.html')


app = Flask(__name__, static_url_path='/')
app.register_blueprint(bp, url_prefix='/')
app.run(host='0.0.0.0', port=5050, debug=True)
