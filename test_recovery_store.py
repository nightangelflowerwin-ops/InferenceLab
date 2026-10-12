from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from chaos_suite import RecoveringCluster, checksum
from recovery_store import RecoveryStore, CorruptState, network_recovery_demo


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'recovery.db'

    def test_restart_recovers_completion_and_checkpoint(self):
        first=RecoveringCluster(store=RecoveryStore(self.path))
        result=first.execute('r','context')
        restarted=RecoveringCluster(store=RecoveryStore(self.path))
        self.assertEqual(restarted.execute('r','context'),result)
        self.assertFalse(restarted.replicas['a'].cache)
        restarted.inject('a','partition')
        result=restarted.execute('new','context')
        self.assertEqual(result['replica'],'b')
        self.assertTrue(result['fixture_cache_hit'])
        with self.assertRaises(ValueError):
            restarted.execute('r','different')

    def test_corrupt_checkpoint_recomputed_but_completion_not_replayed(self):
        cluster=RecoveringCluster(store=RecoveryStore(self.path))
        cluster.execute('r','context')
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('UPDATE checkpoints SET data=? WHERE id=?',('{}',checksum('context')))
            db.commit()
        restarted=RecoveringCluster(store=RecoveryStore(self.path))
        result=restarted.execute('new','context')
        self.assertFalse(result['fixture_cache_hit'])
        self.assertTrue(any(event['event']=='durable_checkpoint_rejected' for event in restarted.events))
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('UPDATE completions SET data=? WHERE id=?',('{}','r'))
            db.commit()
        with self.assertRaises(CorruptState):
            RecoveringCluster(store=RecoveryStore(self.path)).execute('r','context')

    def test_real_socket_interruption_explicit_resume_and_identity_fencing(self):
        result=network_recovery_demo(self.path)
        self.assertTrue(result['completed'])
        self.assertEqual([index for index,_ in result['received_events']],[0,1,2,3,4])
        store=RecoveryStore(self.path)
        with self.assertRaises(RuntimeError):
            store.prepare_stream('network-demo','one two three four')
        with self.assertRaises(ValueError):
            store.prepare_stream('network-demo','different',1)
        self.assertEqual(store.prepare_stream('network-demo','one two three four',4),[])


if __name__=='__main__':
    unittest.main()
