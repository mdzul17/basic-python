import unittest

from ocr_leaderboard import parse_leaderboard_text


class ParseLeaderboardTextTest(unittest.TestCase):
    def test_parses_ranked_rows_from_leaderboard_text(self):
        raw_text = """
        Peringkat Player Power
        6 Raihannnnn ツ 8061
        7 bl@CK Panther™ 7987
        8 Dhika Mediatek 7941
        9 KING EMYU 7890
        10 ellie*2209 7887
        Anda harus memiliki Hero ini untuk memasuki Leaderboard Hero Power.
        """

        entries = parse_leaderboard_text(raw_text, "datasets/leaderboard.jpg")

        self.assertEqual(
            [(entry.rank, entry.nama_pemain, entry.point) for entry in entries],
            [
                (6, "Raihannnnn ツ", 8061),
                (7, "bl@CK Panther™", 7987),
                (8, "Dhika Mediatek", 7941),
                (9, "KING EMYU", 7890),
                (10, "ellie*2209", 7887),
            ],
        )

    def test_parses_unranked_local_player_row(self):
        raw_text = "Tidak Ada Peringkat Gofckyosef Belum Diperoleh"

        entries = parse_leaderboard_text(raw_text, "datasets/leaderboard.jpg")

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].rank)
        self.assertEqual(entries[0].rank_label, "Tidak Ada Peringkat")
        self.assertEqual(entries[0].nama_pemain, "Gofckyosef")
        self.assertIsNone(entries[0].point)
        self.assertEqual(entries[0].point_label, "Belum Diperoleh")

    def test_infers_rank_when_ocr_reads_medal_rows_without_rank_numbers(self):
        raw_text = """
        Peringkat Player + Power
        f iy Ra 4] ZraskOvert ord 10663 a
        [ Oy Ine Sentinel 10150
        & fa Naffx Seger Seger 9654 a
        4 Cy MILLO 9429 a
        """

        entries = parse_leaderboard_text(raw_text, "datasets/leaderboard.jpg")

        self.assertEqual(
            [(entry.rank, entry.nama_pemain, entry.point) for entry in entries],
            [
                (1, "ZraskOvert ord", 10663),
                (2, "Sentinel", 10150),
                (3, "Naffx Seger Seger", 9654),
                (4, "MILLO", 9429),
            ],
        )


if __name__ == "__main__":
    unittest.main()
