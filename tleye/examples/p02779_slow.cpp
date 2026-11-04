#include<bits/stdc++.h>
using namespace std;
int main()
{
  long long int n,i,j;
  cin>>n;
  long long int a[n],b=0;
  for(i=0;i<n;i++)
  {
    cin>>a[i];
  }
  for(i=0;i<n-1;i++)
  {
    for(j=i+1;j<n;j++)
    {
      if(a[i]==a[j])
      {
        b++;
        break;

      }
    }
  }
  if(b==1)
  {

      cout<<"NO";

  }
  else{
    cout<<"YES";

  }
  return 0;
}

