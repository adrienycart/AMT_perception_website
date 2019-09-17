"""add goldmsi

Revision ID: a833844601bd
Revises: 39ea744c2625
Create Date: 2019-09-16 17:25:18.075500

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a833844601bd'
down_revision = '39ea744c2625'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('gold_msi_answer',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('answer1', sa.Integer(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_gold_msi_answer_timestamp'), 'gold_msi_answer', ['timestamp'], unique=False)
    op.add_column(u'answer', sa.Column('example_id', sa.Integer(), nullable=True))
    op.add_column(u'answer', sa.Column('system1_id', sa.Integer(), nullable=True))
    op.add_column(u'answer', sa.Column('system2_id', sa.Integer(), nullable=True))
    op.add_column(u'user', sa.Column('goldmsi_completed', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column(u'user', 'goldmsi_completed')
    op.drop_column(u'answer', 'system2_id')
    op.drop_column(u'answer', 'system1_id')
    op.drop_column(u'answer', 'example_id')
    op.drop_index(op.f('ix_gold_msi_answer_timestamp'), table_name='gold_msi_answer')
    op.drop_table('gold_msi_answer')
    # ### end Alembic commands ###
