""" Verify that the public bucket is readable by strangers and writable by nobody.

Run after applying ``docs/nautilus/s3_llc4320v2_policy.json``::

    wr_llc_v2_check_access s3://llc4320-v2/SURFACE

Every check is made with *unsigned* requests, i.e. as an anonymous visitor.
Reads and listing must pass; writes and deletes must be refused.  Exits
non-zero if any check comes out the wrong way.

The write check uses a scratch key and removes it afterwards; the delete
check targets a key that does not exist, so neither can damage the data.
"""

import argparse
import sys


def parser(options=None):
    p = argparse.ArgumentParser(description='Check anonymous access to the LLC4320 v2 bucket.')
    p.add_argument('root', nargs='?', default='s3://llc4320-v2/SURFACE',
                   help='s3://bucket/prefix (default s3://llc4320-v2/SURFACE)')
    p.add_argument('--endpoint', default='https://s3-west.nrp-nautilus.io')
    return p.parse_args() if options is None else p.parse_args(options)


def main(args):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    bucket, _, prefix = args.root.replace('s3://', '').partition('/')
    prefix = prefix.rstrip('/')
    c = boto3.client('s3', endpoint_url=args.endpoint,
                     config=Config(signature_version=UNSIGNED, s3={'addressing_style': 'path'}))
    results = {}

    def check(name, want_ok, fn):
        try:
            fn()
            ok = True
        except Exception:
            ok = False
        results[name] = (ok, want_ok)
        verdict = 'PASS' if ok == want_ok else 'FAIL'
        print(f"  {verdict}  {name}: {'allowed' if ok else 'denied'} "
              f"(want {'allowed' if want_ok else 'denied'})")

    print(f"anonymous access to s3://{bucket}/{prefix}")
    check('ListBucket', True,
          lambda: c.list_objects_v2(Bucket=bucket, Prefix=prefix + '/', MaxKeys=1))
    check('GetObject (grid metadata)', True,
          lambda: c.get_object(Bucket=bucket, Key=f'{prefix}/grid.zarr/zarr.json')['Body'].read())
    check('PutObject', False,
          lambda: c.put_object(Bucket=bucket, Key=f'{prefix}/_access_check.tmp', Body=b'x'))
    # Deliberately a key that does not exist. S3 returns success for deleting
    # a missing key when the caller is permitted and 403 when it is not, so
    # this distinguishes the two without ever being able to destroy data.
    # (An earlier version of this tool pointed at grid.zarr/zarr.json and
    # deleted it; the group document had to be rebuilt by hand.)
    check('DeleteObject', False,
          lambda: c.delete_object(Bucket=bucket, Key=f'{prefix}/_access_check_absent.tmp'))

    try:
        import xarray as xr
        xr.open_zarr(f's3://{bucket}/{prefix}/grid.zarr', consolidated=None,
                     storage_options={'anon': True,
                                      'client_kwargs': {'endpoint_url': args.endpoint},
                                      'config_kwargs': {'s3': {'addressing_style': 'path'}}})
        ok = True
    except Exception as e:
        ok = False
        print(f"      (xarray error: {type(e).__name__}: {str(e)[:100]})")
    results['xarray open_zarr (anon)'] = (ok, True)
    print(f"  {'PASS' if ok else 'FAIL'}  xarray open_zarr (anon): "
          f"{'works' if ok else 'fails'} (want works)")

    # If PutObject unexpectedly succeeded, clean up after ourselves.
    if results['PutObject'][0]:
        try:
            import boto3 as _b
            from botocore.config import Config as _C
            _b.Session(profile_name='default').client(
                's3', endpoint_url=args.endpoint,
                config=_C(s3={'addressing_style': 'path'})).delete_object(
                    Bucket=bucket, Key=f'{prefix}/_access_check.tmp')
            print("      (removed the object the anonymous write created)")
        except Exception:
            print("      WARNING: could not remove the anonymous test object")

    bad = [k for k, (ok, want) in results.items() if ok != want]
    print("\nall checks passed" if not bad else f"\nFAILED: {', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main(parser()))
