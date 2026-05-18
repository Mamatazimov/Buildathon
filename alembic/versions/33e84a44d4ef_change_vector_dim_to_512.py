"""change_vector_dim_to_512

Revision ID: 33e84a44d4ef
Revises:
Create Date: 2026-05-16 00:46:02.368467

"""

from typing import Sequence, Union

import pgvector
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "33e84a44d4ef"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Eski 128 o'lchamli ustunni o'chiramiz
    op.drop_column("face_vectors", "embedding")

    # 2. 💥 JADVALNI TOZALAYMIZ (Eski o'lik qatorlarni butunlay o'chirib tashlaydi)
    op.execute("TRUNCATE TABLE face_vectors CASCADE;")

    # 3. Endi jadval top-toza, bemalol NOT NULL sharti bilan yangi ustunni qo'shamiz
    op.add_column(
        "face_vectors",
        sa.Column(
            "embedding", pgvector.sqlalchemy.vector.VECTOR(dim=512), nullable=False
        ),
    )


def downgrade() -> None:
    # Orqaga qaytarish mantig'i (shunchaki o'z holatida qolsa ham bo'ladi)
    op.drop_column("face_vectors", "embedding")
    op.add_column(
        "face_vectors",
        sa.Column(
            "embedding", pgvector.sqlalchemy.vector.VECTOR(dim=128), nullable=False
        ),
    )
